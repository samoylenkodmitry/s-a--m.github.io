---
layout: leetcode-entry
title: "3454. Separate Squares II"
permalink: "/leetcode/problem/2026-01-14-3454-separate-squares-ii/"
leetcode_ui: true
entry_slug: "2026-01-14-3454-separate-squares-ii"
---

[3454. Separate Squares II](https://leetcode.com/problems/separate-squares-ii/description/) hard
[blog post](https://leetcode.com/problems/separate-squares-ii/solutions/7493956/kotlin-rust-by-samoylenkodmitry-t19c/)
[substack]()
[youtube](https://youtu.be/TraQ0upCaZE)

![69c10bcc-e8c8-4421-a110-cd3120604cb5 (1).webp](/assets/leetcode_daily_images/966b06b3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1237

#### Problem TLDR

Min y to equal split areas exclude intersections #hard #line_sweep

#### Intuition

Didn't solved.

```j
    // sort by y
    // do a line sweep
    //
    // widths by y
    //
    // 0 0 0 1 1 2 0 0 3 0 0
    //         i             area before & area after
    //       1*2+2*1+3*1     total
    //                       but how to find min y center?
    //
    // ok let's just start with making a widths list
    // how to merge x segments?
    // ok this is hard for me, let's go to hints
    // gave up after 15 minute
```

Line sweep by Y.
Record (x start, x end) pairs.
On each next Y line sweep elapsed x-pairs to find non-overlapping w.

#### Approach

* accepted without segment tree

#### Complexity

- Time complexity:
$$O(nlogn)$$, worst-case would be O(n^2log^2(n))

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 459ms
    fun separateSquares(s: Array<IntArray>): Double {
        val (e,xlr,slices) = List(3) {ArrayList<List<Double>>()}
        for ((x,y,l) in s) {e+=listOf(1.0*y,1.0,1.0*x,1.0*x+l);e+=listOf(1.0*y+l,0.0,1.0*x,1.0*x+l)}
        e.sortBy { it[0] }; var prev = 0.0; var total = 0.0
        for ((y,e,xl,xr) in e) {
            if (y > prev) {
                xlr.sortBy {it[0]}; var w = 0.0; var x = 0.0
                for ((l,r) in xlr) if (x<r) { w += r - max(x,l); x = r }
                slices += listOf(total, prev, y-prev, w); total += (y-prev)*w
            }
            prev = y; if (e>0) xlr += listOf(xl,xr) else xlr -= listOf(xl,xr)
        }
        val (cur,sy,h,w) = slices.first { (cur, sy,h,w) -> cur + h * w >= total/2  }
        return (total/2 - cur) / w + sy
    }
```
```rust
// 373ms
    pub fn separate_squares(s: Vec<Vec<i32>>) -> f64 {
        let mut ev: Vec<_> = s.iter().flat_map(|v| {
        let (x, y, l) = (v[0] as f64, v[1] as f64, v[2] as f64); [(y,1,x,x+l),(y+l,-1,x,x+l)]}).collect();
        ev.sort_by(|a, b| a.0.total_cmp(&b.0)); let (mut xlr, mut sl, mut tot, mut py) = (vec![], vec![], 0., 0.);
        for (y, op, l, r) in ev {
            if y > py {
                xlr.sort_by(|a: &(f64, f64), b| a.0.total_cmp(&b.0));
                let (mut w, mut mx) = (0., 0.);
                for &(al, ar) in &xlr { if mx < ar { w += ar - mx.max(al); mx = ar}}
                sl.push((tot, py, y - py, w)); tot += (y - py) * w; py = y
            }
            if op > 0 { xlr.push((l, r)); }
            else { xlr.remove(xlr.iter().position(|&x| x == (l, r)).unwrap()); }
        }
        let (cur, y, _, w) = sl.into_iter().find(|&(c, _, h, w)| c + h * w >= tot / 2.).unwrap();
        (tot / 2. - cur) / w + y
    }
```

