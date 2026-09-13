---
layout: leetcode-entry
title: "835. Image Overlap"
permalink: "/leetcode/problem/2026-09-13-835-image-overlap/"
leetcode_ui: true
entry_slug: "2026-09-13-835-image-overlap"
---

[835. Image Overlap](https://leetcode.com/problems/image-overlap/solutions/8519260/kotlin-rust-by-samoylenkodmitry-4lkj/) medium
[substack](https://dmitriisamoilenko.substack.com/p/13092026-835-image-overlap?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ypaM6wU5pqs)

https://dmitrysamoylenko.com/leetcode/

![13.09.2026.webp](/assets/leetcode_daily_images/13.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1481

#### Problem TLDR

Best intersection after shift 2D matrix

#### Intuition

Brute-force. Check all the shifts.

#### Approach

* or group by shift vectors (dx,dy) between points

#### Complexity

- Time complexity:
$$O(n^4)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun largestOverlap(a: Array<IntArray>, b: Array<IntArray>)=
        (-29..29).run{maxOf{maxOf{u->sumOf{y->sumOf{x->
            try{b[y][x]*a[y+it][x+u]}catch(e:Exception){0}}}}}}
```
```rust
    pub fn largest_overlap(a: Vec<Vec<i32>>, b: Vec<Vec<i32>>) -> i32 {
        let n=a.len();*iproduct!(0..n,0..n,0..n,0..n).filter(|&(r,c,s,d)|a[r][c]*b[s][d]>0)
        .counts_by(|(r,c,s,d)|(r+29-s,c+29-d)).values().max().unwrap_or(&0)as _
    }
```

