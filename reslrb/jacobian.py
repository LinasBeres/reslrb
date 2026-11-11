import drjit as dr
import mitsuba as mi


class TwoPointParametrization:
    """A ray segment written as the barycentric coordinates of its two endpoints."""

    def __init__(self, bary_start, start_valid, bary_end, end_valid, wo):
        self.uv_start = dr.select(start_valid, bary_start, mi.Point2f(0.0, 0.0))
        self.uv_end = dr.select(end_valid, bary_end, mi.warp.uniform_sphere_to_square(wo))

    def params_list(self):
        return [self.uv_start.x, self.uv_start.y, self.uv_end.x, self.uv_end.y]

    def spawn_ray(self, si_start, si_end, ray=dr.zeros(mi.Ray3f)):
        p0 = triangle_point(si_start, self.uv_start)
        frame = mi.Frame3f(ray.d)
        p0[~si_start.is_valid()] = ray.o + frame.s * self.uv_start.x + frame.t * self.uv_start.y

        p1 = triangle_point(si_end, self.uv_end)
        p1[~si_end.is_valid()] = p0 + mi.warp.square_to_uniform_sphere(self.uv_end)

        direction = p1 - p0
        d = direction / dr.norm(direction)
        mag = (1.0 + dr.max(dr.abs(p0))) * mi.math.RayEpsilon
        mag = dr.detach(dr.mulsign(mag, dr.dot(si_start.n, d)))
        origin = mag * dr.detach(si_start.n) + p0
        return mi.Ray3f(origin, d, si_start.time, si_start.wavelengths)


def triangle_point(si, uv):
    mesh = mi.MeshPtr(si.shape)
    fi = mesh.face_indices(si.prim_index)
    p0 = mesh.vertex_position(fi[0])
    p1 = mesh.vertex_position(fi[1])
    p2 = mesh.vertex_position(fi[2])
    return (1 - uv.x - uv.y) * p0 + uv.x * p1 + uv.y * p2


def safe_inverse(M, eps=1e-8):
    """Inverse of M, or the zero matrix where M is singular."""
    safe = dr.abs(dr.det(M)) > eps
    inv = dr.inverse(M + dr.select(safe, 0.0, 1.0) * mi.Matrix4f(1.0))
    return dr.select(safe, inv, mi.Matrix4f(0.0))


def jacobian(params, output):
    """Forward-mode Jacobian of a 4D output with respect to the four ray parameters."""
    result = mi.Matrix4f(0.0)
    for i in range(len(params)):
        dr.set_grad(output, 0.0)
        last = i == len(params) - 1
        flags = dr.ADFlag.Default if last else dr.ADFlag.ClearInput | dr.ADFlag.ClearInterior
        dr.forward(params[i], flags=flags)
        grad = dr.grad(output)
        result[0, i] = grad.x
        result[1, i] = grad.y
        result[2, i] = grad.z
        result[3, i] = grad.w
    return result


def barycentric(pi, ray):
    """Barycentric coordinates of the hit, recomputed so they carry derivatives of the ray."""
    mesh = mi.MeshPtr(pi.shape)
    fi = mesh.face_indices(pi.prim_index)
    p0 = mesh.vertex_position(fi[0])
    p1 = mesh.vertex_position(fi[1])
    p2 = mesh.vertex_position(fi[2])

    e1 = p1 - p0
    e2 = p2 - p0
    pvec = dr.cross(ray.d, e2)
    inv_det = dr.rcp(dr.dot(e1, pvec))
    tvec = ray.o - p0
    u = dr.dot(tvec, pvec) * inv_det
    qvec = dr.cross(tvec, e1)
    v = dr.dot(ray.d, qvec) * inv_det
    t = dr.dot(e2, qvec) * inv_det

    hit = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0) & (t <= ray.maxt)
    return dr.select(hit, mi.Point2f(u, v), pi.prim_uv)
