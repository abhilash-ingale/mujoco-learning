"""Stream a running MuJoCo simulation to a browser with viser.

MuJoCo's own viewer is an OpenGL window bound to the local machine's main
thread. `viser` instead runs a small web server and pushes geometry over a
websocket, so the same simulation is viewable in any browser -- including on a
remote box, over SSH port-forwarding, or on a headless server where
`mujoco.viewer` cannot open a window at all.

The bridge is deliberately thin, and rests on one observation: after every
`mj_step`, MuJoCo has already resolved the world pose of every geom into

    data.geom_xpos[gid]   (3,)  position
    data.geom_xmat[gid]   (9,)  rotation, row-major 3x3

So the split is:

  ONCE   upload each geom's mesh (vertices + faces) to the browser
  EVERY FRAME  send only the 7 numbers of each geom's pose

Uploading ~200k vertices once and then streaming poses is what keeps this
usable at interactive rates; re-uploading meshes per frame would not be.

Used by g1_walk.py via `--viser`, but it is model-agnostic -- point it at any
MjModel/MjData pair.
"""

from __future__ import annotations

import time

import numpy as np
import trimesh
import viser

import mujoco

# MuJoCo convention: geom_group 3 holds collision primitives, group 2 the visual
# meshes. Rendering both gives you the robot wearing its own hitboxes.
COLLISION_GROUP = 3


def geom_mesh(model: mujoco.MjModel, gid: int) -> trimesh.Trimesh | None:
    """Local-frame triangle mesh for one geom, or None if it has no geometry.

    Mesh geoms carry their vertices in the model (already scaled at compile
    time); primitives get tessellated by trimesh.
    """
    gtype, size = model.geom_type[gid], model.geom_size[gid]

    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        return trimesh.Trimesh(
            vertices=model.mesh_vert[v0 : v0 + nv],
            faces=model.mesh_face[f0 : f0 + nf],
        )
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size[:3])
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        # MuJoCo stores the HALF-length in size[1]; trimesh wants full height.
        return trimesh.creation.cylinder(radius=float(size[0]), height=2 * float(size[1]))
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        # trimesh builds capsules from z=0 upward, MuJoCo centres them on the
        # geom origin -- so recentre, or every capsule sits half a body too high.
        cap = trimesh.creation.capsule(radius=float(size[0]), height=2 * float(size[1]))
        cap.apply_translation((0.0, 0.0, -float(size[1])))
        return cap
    # Planes are infinite and have no mesh; drawn as a viser grid instead.
    return None


def geom_color(model: mujoco.MjModel, gid: int) -> tuple[tuple[int, int, int], float]:
    """RGB 0-255 and opacity for a geom, preferring its material over geom_rgba."""
    matid = model.geom_matid[gid]
    rgba = model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[gid]
    return tuple(int(255 * c) for c in rgba[:3]), float(rgba[3])


class ViserBridge:
    """Mirrors an MjData into a browser scene. Call `sync()` after `mj_step`."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        port: int = 8080,
        show_collision: bool = False,
        follow: bool = False,
    ) -> None:
        self.model, self.data, self.follow = model, data, follow
        self._next_cam = 0.0
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")  # MuJoCo is z-up; three.js is y-up
        self.server.scene.add_grid("/floor", width=40, height=40, cell_size=0.5)

        # --- upload geometry once ---
        self._handles: list[tuple[int, viser.MeshHandle]] = []
        for gid in range(model.ngeom):
            if not show_collision and model.geom_group[gid] == COLLISION_GROUP:
                continue
            mesh = geom_mesh(model, gid)
            if mesh is None:
                continue
            color, opacity = geom_color(model, gid)
            handle = self.server.scene.add_mesh_simple(
                name=f"/geom/{gid}",
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.uint32),
                color=color,
                opacity=None if opacity >= 1.0 else opacity,
                flat_shading=False,
            )
            self._handles.append((gid, handle))

        self._quat = np.zeros(4)  # scratch buffer for mju_mat2Quat
        self._t_label = self.server.gui.add_text("sim time", initial_value="0.00 s")
        print(f"viser: open http://localhost:{port}  ({len(self._handles)} geoms)")

    def sync(self) -> None:
        """Push the current MjData pose of every geom to connected browsers.

        `server.atomic()` batches the whole frame into one websocket message;
        without it each geom would be sent separately and the robot would visibly
        tear apart mid-update.
        """
        data, quat = self.data, self._quat
        with self.server.atomic():
            for gid, handle in self._handles:
                handle.position = data.geom_xpos[gid]
                mujoco.mju_mat2Quat(quat, data.geom_xmat[gid])
                handle.wxyz = quat
            self._t_label.value = f"{data.time:.2f} s"
        # Camera-follow is OFF by default and throttled to 4 Hz when on. Writing
        # client.camera.look_at at frame rate fights the browser's own orbit
        # controls -- it makes the view impossible to drag and can wedge the
        # renderer outright. Learned the hard way; leave the camera alone.
        if self.follow:
            now = time.perf_counter()
            if now >= self._next_cam:
                self._next_cam = now + 0.25
                for client in self.server.get_clients().values():
                    client.camera.look_at = data.qpos[:3]

    def wait_for_client(self, timeout: float = 30.0) -> bool:
        """Block until a browser connects, so the run isn't over before you look."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.server.get_clients():
                return True
            time.sleep(0.1)
        return False
