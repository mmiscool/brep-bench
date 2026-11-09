//! Newell's teapot, with Crow's bottom, in Newell's coordinates: the union of four solids, each
//! bounded by bicubic Bézier patches and closed by flat faces.
//!
//! next.BREP has no operation that makes a face from a surface, so each solid is stated in the
//! kernel's own topology records (`solids::geometry::Records`) and validated by the kernel: every
//! patch is one face on its own Bézier surface; a boundary two patches share is one edge, the
//! same control polygon either way round; a boundary collapsed to one point is a pole. Mirror
//! images reverse a patch's sense, so the patches are given one sense by walking their shared
//! edges, and the solid's sign by the kernel's volume. What the patches leave open is closed by
//! flat faces, one per open loop, on the plane the loop lies in. The kernel's boolean unites the
//! four.
use crate::solids::boolean::union;
use crate::solids::geometry::{uv_line, Records};
use crate::solids::p;
use crate::Refusal;
use brep_kernel::{make_plane, solid_mass_properties, BrepSolid, NurbsCurve, NurbsSurface, Vec4};
use std::collections::HashMap;

type P = [f64; 3];
/// A patch's control points, row i (u) by column j (v).
type Net = [[P; 4]; 4];

/// A ring's first patch, the quarter from +x to −y, from its profile of four (r, z) points.
fn quarter(profile: [(f64, f64); 4]) -> Net {
    profile.map(|(r, z)| [[r, 0.0, z], [r, -0.56 * r, z], [0.56 * r, -r, z], [0.0, -r, z]])
}

/// `net` turned a quarter about the z axis, exactly: (x, y) to (−y, x).
fn turned(net: &Net) -> Net {
    net.map(|row| row.map(|[x, y, z]| [-y, x, z]))
}

/// The ring of four patches: the quarter and it turned 90°, 180° and 270° about the z axis.
fn ring(first: Net) -> Vec<Net> {
    let mut patches = vec![first];
    for _ in 0..3 {
        let next = turned(patches.last().unwrap());
        patches.push(next);
    }
    patches
}

/// `net` and its mirror image in the plane y = 0.
fn mirrored(net: Net) -> [Net; 2] {
    [net, net.map(|row| row.map(|[x, y, z]| [x, -y, z]))]
}

fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: P, b: P) -> P {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: P, b: P) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn unit(a: P) -> P {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

/// A point as an exact key: the inputs are decimal constants turned and mirrored exactly, so equal
/// points have equal bits (up to the sign of a zero).
fn key(point: P) -> [u64; 3] {
    point.map(|c| (c + 0.0).to_bits())
}

/// The four sides of a patch, running round its parameter square counter-clockwise: the side
/// v = 0 with u rising, u = 1 with v rising, v = 1 with u falling, u = 0 with v falling. Each is
/// its four control points in the order run, and its pcurve's ends.
fn sides(net: &Net) -> [([P; 4], [f64; 4]); 4] {
    let column = |j: usize| [net[0][j], net[1][j], net[2][j], net[3][j]];
    let rev = |s: [P; 4]| [s[3], s[2], s[1], s[0]];
    [
        (column(0), [0.0, 0.0, 1.0, 0.0]),
        (net[3], [1.0, 0.0, 1.0, 1.0]),
        (rev(column(3)), [1.0, 1.0, 0.0, 1.0]),
        (rev(net[0]), [0.0, 1.0, 0.0, 0.0]),
    ]
}

fn degenerate(side: &[P; 4]) -> bool {
    side.iter().all(|&q| key(q) == key(side[0]))
}

/// `net` with its two parameters exchanged: the same surface, its normal Su × Sv reversed.
fn transposed(net: &Net) -> Net {
    let mut out = *net;
    for (i, row) in out.iter_mut().enumerate() {
        for (j, q) in row.iter_mut().enumerate() {
            *q = net[j][i];
        }
    }
    out
}

/// A side's key as an undirected edge, and whether it runs the key's way.
fn edge_key(side: &[P; 4]) -> (Vec<[u64; 3]>, bool) {
    let forward: Vec<_> = side.iter().map(|&q| key(q)).collect();
    let mut backward = forward.clone();
    backward.reverse();
    if forward <= backward { (forward, true) } else { (backward, false) }
}

/// The patches given one sense: each patch whose shared side runs the same way as its neighbour's
/// is transposed, walking outward from the first.
fn one_sense(patches: &[Net]) -> Result<Vec<Net>, String> {
    let mut users: HashMap<Vec<[u64; 3]>, Vec<(usize, bool)>> = HashMap::new();
    for (index, net) in patches.iter().enumerate() {
        for (side, _) in sides(net) {
            if !degenerate(&side) {
                let (k, along) = edge_key(&side);
                users.entry(k).or_default().push((index, along));
            }
        }
    }
    if let Some((_, u)) = users.iter().find(|(_, u)| u.len() > 2) {
        return Err(format!("a patch side is shared by {} patches", u.len()));
    }
    let mut flip: Vec<Option<bool>> = vec![None; patches.len()];
    let mut stack = vec![0];
    flip[0] = Some(false);
    while let Some(index) = stack.pop() {
        let own = flip[index].unwrap();
        for (side, _) in sides(&patches[index]) {
            if degenerate(&side) {
                continue;
            }
            let (k, along) = edge_key(&side);
            for &(other, other_along) in &users[&k] {
                if other == index {
                    continue;
                }
                // Opposite runs once both senses are applied: a transpose reverses every run.
                let wanted = (along ^ own) == (other_along ^ true);
                let other_flip = if wanted { false } else { true };
                match flip[other] {
                    None => {
                        flip[other] = Some(other_flip);
                        stack.push(other);
                    }
                    Some(f) if f != other_flip => {
                        return Err("the patches cannot be given one sense".into())
                    }
                    _ => {}
                }
            }
        }
    }
    patches
        .iter()
        .zip(flip)
        .map(|(net, f)| match f {
            Some(true) => Ok(transposed(net)),
            Some(false) => Ok(*net),
            None => Err("the patches are not connected".into()),
        })
        .collect()
}

fn bezier(points: &[P; 4]) -> Result<NurbsCurve, String> {
    let control = points.iter().map(|&q| Vec4::from_point(p(q[0], q[1], q[2]), 1.0)).collect();
    NurbsCurve::new(3, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], control)
}

fn surface(net: &Net) -> Result<NurbsSurface, String> {
    let rows = net
        .iter()
        .map(|row| row.iter().map(|&q| Vec4::from_point(p(q[0], q[1], q[2]), 1.0)).collect())
        .collect();
    let cubic = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    NurbsSurface::new(3, 3, cubic.clone(), cubic, rows)
}

/// The solid bounded by `patches` (given one sense) and by a flat face over each loop they leave
/// open; `reversed` turns every patch over.
fn closed(patches: &[Net], reversed: bool) -> Result<BrepSolid, String> {
    let patches: Vec<Net> =
        patches.iter().map(|net| if reversed { transposed(net) } else { *net }).collect();
    let mut b = Records::new();
    let mut vertices = HashMap::new();
    let mut vertex = |b: &mut Records, q: P| {
        *vertices.entry(key(q)).or_insert_with(|| b.vertex(p(q[0], q[1], q[2])))
    };
    // Each undirected edge, its record and the way its curve runs; and the sides used once.
    let mut edges = HashMap::new();
    let mut uses: HashMap<Vec<[u64; 3]>, usize> = HashMap::new();
    for net in &patches {
        for (side, _) in sides(net) {
            if !degenerate(&side) {
                *uses.entry(edge_key(&side).0).or_default() += 1;
            }
        }
    }
    let mut open: Vec<[P; 4]> = Vec::new();
    for net in &patches {
        let mut coedges = Vec::new();
        for (side, [u0, v0, u1, v1]) in sides(net) {
            let pcurve = uv_line(u0, v0, u1, v1)?;
            if degenerate(&side) {
                let at = vertex(&mut b, side[0]);
                coedges.push((b.pole(at)?, true, pcurve));
                continue;
            }
            let (k, along) = edge_key(&side);
            if uses[&k] == 1 {
                open.push(side);
            }
            let edge = match edges.get(&k) {
                Some(&edge) => edge,
                None => {
                    let run = if along { side } else { [side[3], side[2], side[1], side[0]] };
                    let (start, end) = (vertex(&mut b, run[0]), vertex(&mut b, run[3]));
                    let edge = b.edge(bezier(&run)?, start, end)?;
                    edges.insert(k.clone(), edge);
                    edge
                }
            };
            coedges.push((edge, along, pcurve));
        }
        b.face(surface(net)?, coedges);
    }
    // The open sides chained into loops, each run backwards, as the flat face runs them.
    while let Some(first) = open.pop() {
        let mut lap = vec![first];
        loop {
            let tail = lap.last().unwrap()[0];
            if key(tail) == key(lap[0][3]) {
                break;
            }
            let next = open
                .iter()
                .position(|s| key(s[3]) == key(tail))
                .ok_or("an open loop does not close")?;
            lap.push(open.remove(next));
        }
        // The loop's plane: its vector area (as the flat face runs it) is the outward normal.
        let samples: Vec<P> = lap
            .iter()
            .flat_map(|s| {
                let curve = bezier(&[s[3], s[2], s[1], s[0]]).unwrap();
                (0..16).map(move |k| {
                    let q = curve.evaluate(k as f64 / 16.0).unwrap();
                    [q.x, q.y, q.z]
                })
            })
            .collect();
        let n = samples.len() as f64;
        let centre = samples.iter().fold([0.0; 3], |c, q| [c[0] + q[0] / n, c[1] + q[1] / n, c[2] + q[2] / n]);
        let mut area = [0.0; 3];
        for k in 0..samples.len() {
            let c = cross(sub(samples[k], centre), sub(samples[(k + 1) % samples.len()], centre));
            area = [area[0] + c[0], area[1] + c[1], area[2] + c[2]];
        }
        let normal = unit(area);
        let helper = if normal[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let ex = unit(cross(helper, normal));
        let ey = cross(normal, ex);
        let reach = samples.iter().map(|&q| dot(sub(q, centre), sub(q, centre)).sqrt()).fold(0.0, f64::max);
        let half = 1.5 * reach;
        let origin = [
            centre[0] - half * (ex[0] + ey[0]),
            centre[1] - half * (ex[1] + ey[1]),
            centre[2] - half * (ex[2] + ey[2]),
        ];
        let plane = make_plane(
            p(origin[0], origin[1], origin[2]),
            p(ex[0], ex[1], ex[2]),
            p(ey[0], ey[1], ey[2]),
            2.0 * half,
            2.0 * half,
        )?;
        let mut run = Vec::new();
        for s in &lap {
            let (k, along) = edge_key(s);
            run.push((edges[&k], !along));
        }
        b.flat(plane, &run)?;
    }
    b.solid(0)
}

/// The solid bounded by `patches` and the flat faces over their open loops, turned outward.
fn solid(patches: &[Net]) -> Result<BrepSolid, String> {
    let patches = one_sense(patches)?;
    let body = closed(&patches, false)?;
    if solid_mass_properties(&body)?.volume > 0.0 {
        Ok(body)
    } else {
        closed(&patches, true)
    }
}

pub fn build() -> Result<BrepSolid, Refusal> {
    let mut knob = quarter([(0.0, 3.15), (0.8, 3.15), (0.0, 2.85), (0.2, 2.7)]);
    knob[1] = [[0.8, 0.0, 3.15], [0.8, -0.45, 3.15], [0.45, -0.8, 3.15], [0.0, -0.8, 3.15]];

    let body_patches: Vec<Net> = [
        quarter([(1.4, 2.4), (1.3375, 2.53125), (1.4375, 2.53125), (1.5, 2.4)]),
        quarter([(1.5, 2.4), (1.75, 1.875), (2.0, 1.35), (2.0, 0.9)]),
        quarter([(2.0, 0.9), (2.0, 0.45), (1.5, 0.225), (1.5, 0.15)]),
        quarter([(1.5, 0.15), (1.5, 0.075), (1.425, 0.0), (0.0, 0.0)]),
    ]
    .into_iter()
    .flat_map(ring)
    .collect();
    let lid_patches: Vec<Net> =
        [knob, quarter([(0.2, 2.7), (0.4, 2.55), (1.3, 2.55), (1.3, 2.4)])].into_iter().flat_map(ring).collect();

    let spout_lower: Net = [
        [[1.7, 0.0, 1.425], [1.7, -0.66, 1.425], [1.7, -0.66, 0.6], [1.7, 0.0, 0.6]],
        [[2.6, 0.0, 1.425], [2.6, -0.66, 1.425], [3.1, -0.66, 0.825], [3.1, 0.0, 0.825]],
        [[2.3, 0.0, 2.1], [2.3, -0.25, 2.1], [2.4, -0.25, 2.025], [2.4, 0.0, 2.025]],
        [[2.7, 0.0, 2.4], [2.7, -0.25, 2.4], [3.3, -0.25, 2.4], [3.3, 0.0, 2.4]],
    ];
    let spout_lip: Net = [
        [[2.7, 0.0, 2.4], [2.7, -0.25, 2.4], [3.3, -0.25, 2.4], [3.3, 0.0, 2.4]],
        [[2.8, 0.0, 2.475], [2.8, -0.25, 2.475], [3.525, -0.25, 2.49375], [3.525, 0.0, 2.49375]],
        [[2.9, 0.0, 2.475], [2.9, -0.15, 2.475], [3.45, -0.15, 2.5125], [3.45, 0.0, 2.5125]],
        [[2.8, 0.0, 2.4], [2.8, -0.15, 2.4], [3.2, -0.15, 2.4], [3.2, 0.0, 2.4]],
    ];
    let handle_upper: Net = [
        [[-1.6, 0.0, 2.025], [-1.6, -0.3, 2.025], [-1.5, -0.3, 2.25], [-1.5, 0.0, 2.25]],
        [[-2.3, 0.0, 2.025], [-2.3, -0.3, 2.025], [-2.5, -0.3, 2.25], [-2.5, 0.0, 2.25]],
        [[-2.7, 0.0, 2.025], [-2.7, -0.3, 2.025], [-3.0, -0.3, 2.25], [-3.0, 0.0, 2.25]],
        [[-2.7, 0.0, 1.8], [-2.7, -0.3, 1.8], [-3.0, -0.3, 1.8], [-3.0, 0.0, 1.8]],
    ];
    let handle_lower: Net = [
        [[-2.7, 0.0, 1.8], [-2.7, -0.3, 1.8], [-3.0, -0.3, 1.8], [-3.0, 0.0, 1.8]],
        [[-2.7, 0.0, 1.575], [-2.7, -0.3, 1.575], [-3.0, -0.3, 1.35], [-3.0, 0.0, 1.35]],
        [[-2.5, 0.0, 1.125], [-2.5, -0.3, 1.125], [-2.65, -0.3, 0.9375], [-2.65, 0.0, 0.9375]],
        [[-2.0, 0.0, 0.9], [-2.0, -0.3, 0.9], [-1.9, -0.3, 0.6], [-1.9, 0.0, 0.6]],
    ];
    let spout_patches: Vec<Net> = [spout_lower, spout_lip].into_iter().flat_map(mirrored).collect();
    let handle_patches: Vec<Net> = [handle_upper, handle_lower].into_iter().flat_map(mirrored).collect();

    let body = solid(&body_patches)?;
    let lid = solid(&lid_patches)?;
    let spout = solid(&spout_patches)?;
    let handle = solid(&handle_patches)?;
    let pot = union(&body, &lid)?;
    let pot = union(&pot, &spout)?;
    Ok(union(&pot, &handle)?)
}
