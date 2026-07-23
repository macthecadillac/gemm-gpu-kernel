#[inline]
fn kernel<'a>(
    ai: Strides,
    bi: Strides,
    ci: Strides,
    a: &'a Matrix,
    b: &'a Matrix,
    c: &'a mut Matrix,
    alpha: f64,
) {
    let m = ai.i_end - ai.i_start;
    let l = ai.j_end - ai.j_start;
    let n = bi.j_end - bi.j_start;

    // Pack B as NR-column micro-panels. Within each panel, advancing K
    // advances to the next contiguous group of NR values.
    let mut packed_b = Vec::with_capacity(l * n);
    for j in (0..n).step_by(NR) {
        for k in 0..l {
            let src_start = (bi.i_start + k) * bi.n + bi.j_start + j;
            packed_b.extend_from_slice(&b.data[src_start..src_start + NR]);
        }
    }

    // Reuse one packed A micro-panel for each MR-row block.
    let mut packed_a = vec![0.; l * MR];
    for i in (0..m).step_by(MR) {
        let a_start = (ai.i_start + i) * ai.n + ai.j_start;
        let c_start = (ci.i_start + i) * ci.n + ci.j_start;
        let c_rows = c.data.get_disjoint_mut([
            c_start..c_start + n,
            c_start + ci.n..c_start + n + ci.n,
            c_start + 2 * ci.n..c_start + n + 2 * ci.n,
            c_start + 3 * ci.n..c_start + n + 3 * ci.n
        ]).unwrap();

        // Pack A as K-major groups of MR rows and apply alpha once.
        for k in 0..l {
            for row in 0..MR {
                packed_a[k * MR + row] =
                    alpha * a.data[a_start + k + row * ai.n];
            }
        }

        for j in (0..n).step_by(NR) {
            let mut c_tile = [[0.; NR]; MR];
            for row in 0..MR {
                c_tile[row].copy_from_slice(&c_rows[row][j..j + NR]);
            }

            let panel_start = (j / NR) * l * NR;
            for k in 0..l {
                let b_start = panel_start + k * NR;
                let b_values = &packed_b[b_start..b_start + NR];

                for row in 0..MR {
                    let a_ik = packed_a[k * MR + row];
                    c_tile[row][0] = a_ik.mul_add(b_values[0], c_tile[row][0]);
                    c_tile[row][1] = a_ik.mul_add(b_values[1], c_tile[row][1]);
                    c_tile[row][2] = a_ik.mul_add(b_values[2], c_tile[row][2]);
                    c_tile[row][3] = a_ik.mul_add(b_values[3], c_tile[row][3]);
                    c_tile[row][4] = a_ik.mul_add(b_values[4], c_tile[row][4]);
                    c_tile[row][5] = a_ik.mul_add(b_values[5], c_tile[row][5]);
                    c_tile[row][6] = a_ik.mul_add(b_values[6], c_tile[row][6]);
                    c_tile[row][7] = a_ik.mul_add(b_values[7], c_tile[row][7]);
                }
            }

            for row in 0..MR {
                c_rows[row][j..j + NR].copy_from_slice(&c_tile[row]);
            }
        }
    };
}
