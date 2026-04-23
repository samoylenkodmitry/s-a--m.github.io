---
layout: leetcode-entry
title: "3453. Separate Squares I"
permalink: "/leetcode/problem/2026-01-13-3453-separate-squares-i/"
leetcode_ui: true
entry_slug: "2026-01-13-3453-separate-squares-i"
---

[3453. Separate Squares I](https://leetcode.com/problems/separate-squares-i/description/) medium
[blog post](https://leetcode.com/problems/separate-squares-i/solutions/7491462/kotlin-rust-by-samoylenkodmitry-faon/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12012026-3453-separate-squares-i?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2YzSUpeFXu4)

![ff3c746d-2626-425a-bc74-3098fa8a7666 (1).webp](/assets/leetcode_daily_images/67ebe4e7.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1236

#### Problem TLDR

Min Y to split into equal areas #medium #bs

#### Intuition

```j
    // 1. binary search, but with doubles
    // 2. idk about any other ideas
    // 3. why do we need X coordinate?
    //
    // binary search gives wrong result on big numbers
    // ok 22 minute, lets read hints - "binary search"
    // so, no extra hints
    // overflow
    //
    // ok 50 minutes, let's gave up
```

Binary search on Y, compare the below and above areas.

#### Approach

* carefull with overflow `l*l`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 94ms
    fun separateSquares(s: Array<IntArray>): Double {
        var lo = 0.0; var hi = 1000000000.0; val t = s.sumOf {1.0*it[2]*it[2]}/2
        while (abs(lo - hi) > 0.00001) {
            val m = lo + (hi - lo) / 2.0
            if (s.sumOf{(x,y,l)->if(m>y)min(m-y,1.0*l)*l else 0.0}>=t) hi = m else lo = m
        }
        return lo
    }
```
```rust
// 96ms
    pub fn separate_squares(s: Vec<Vec<i32>>) -> f64 {
        let (mut l, mut h, t) = (0.,1e9, s.iter().map(|v| v[2]as f64 * v[2] as f64).sum::<f64>()/2.);
        while h - l > 1e-5 {
            let m = (l + h) / 2.;
            if t > s.iter().map(|v| {
                    let (y,s) = (v[1] as f64, v[2] as f64);
                    if m > y { (m - y).min(s) * s } else { 0. }
                }).sum::<f64>() { l = m } else { h = m }
        } l
    }
```

