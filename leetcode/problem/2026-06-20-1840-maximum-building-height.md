---
layout: leetcode-entry
title: "1840. Maximum Building Height"
permalink: "/leetcode/problem/2026-06-20-1840-maximum-building-height/"
leetcode_ui: true
entry_slug: "2026-06-20-1840-maximum-building-height"
---

[1840. Maximum Building Height](https://leetcode.com/problems/maximum-building-height/solutions/8346747/kotlin-rust-by-samoylenkodmitry-tau9/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20062026-1840-maximum-building-height?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/To5TLl_lvGQ)

https://dmitrysamoylenko.com/leetcode/

![20.06.2026.webp](/assets/leetcode_daily_images/20.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1396

#### Problem TLDR

Max growing height with restrictions

#### Intuition

```j
    // 123456789 10 11
    // 123456543 21
    // 1    6  3

    // 1 2 3 4 5 6 7 8 9 10
    // 0 5     3   4     3
    //
    //         some restriction in the middle can backtrack
    //
    // 30 minutes, hints: two passes
    // 48 minute: my formula is wrong
    // 56 minute: give up
```
* adjust restrictions with forward anb backward passes
* for each restriction interval the max height = (L+R+d)/2, solve 2D geometry

#### Approach

* on backward pass we already can calculte the max
* we can use 'previous' height and solve in O(1) memory, or use helper collection with 0 and n-th positions

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n|1)$$

#### Code

```kotlin
    fun maxBuilding(n: Int, r: Array<IntArray>): Int {
        r.sortBy { it[0] }; val l = arrayOf(intArrayOf(1, 0)) + r + intArrayOf(n, n - 1)
        for (k in 1..l.lastIndex) l[k][1] = min(l[k][1], l[k-1][1] + l[k][0] - l[k-1][0])
        return (l.lastIndex downTo 1).maxOf { k ->
            val (i, h) = l[k]; val (j, p) = l[k-1]; l[k-1][1] = minOf(p, h + i - j)
            (i - j + h + l[k-1][1]) / 2
        }
    }
```
```rust
    pub fn max_building(n: i32, mut r: Vec<Vec<i32>>) -> i32 {
        r.sort(); let mut l = [vec![vec![1, 0]], r, vec![vec![n, n - 1]]].concat();
        for k in 1..l.len() { l[k][1] = l[k][1].min(l[k-1][1] + l[k][0] - l[k-1][0]); }
        (1..l.len()).rev().fold(0, |res, k| {
            let (i, h, j, p) = (l[k][0], l[k][1], l[k-1][0], l[k-1][1]);
            l[k-1][1] = p.min(h + i - j);
            res.max((i - j + h + l[k-1][1]) / 2)
        })
    }
```

