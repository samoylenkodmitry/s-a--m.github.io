---
layout: leetcode-entry
title: "3047. Find the Largest Area of Square Inside Two Rectangles"
permalink: "/leetcode/problem/2026-01-17-3047-find-the-largest-area-of-square-inside-two-rectangles/"
leetcode_ui: true
entry_slug: "2026-01-17-3047-find-the-largest-area-of-square-inside-two-rectangles"
---

[3047. Find the Largest Area of Square Inside Two Rectangles](https://leetcode.com/problems/find-the-largest-area-of-square-inside-two-rectangles/description) medium
[blog post](https://leetcode.com/problems/find-the-largest-area-of-square-inside-two-rectangles/solutions/7501758/kotlin-rust-by-samoylenkodmitry-urr8/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17012026-3047-find-the-largest-area?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8EK9UqcqPDo)

![977a91cb-099f-459f-a51a-e709a9e03c5e (1).webp](/assets/leetcode_daily_images/f38a5f13.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1240

#### Problem TLDR

Max square in intersection #medium

#### Intuition

Brute-force is accepted.
The intersection is max(bottom left) & min(top right)

#### Approach

* we also can sort and return inner loop early
* or freeze the result S length and binary search if it fits
* or sort and line sweep 2d or with segment tree

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 350ms
    fun largestSquareArea(a: Array<IntArray>, b: Array<IntArray>) =
        (0..<b.size-1).maxOf { i -> (i+1..<b.size).maxOf { j ->
            (0..1).minOf { min(b[i][it],b[j][it])-max(a[i][it],a[j][it]) }
        }}.let { 1L*it*max(0,it) }
```
```rust
// 77ms
    pub fn largest_square_area(a: Vec<Vec<i32>>, b: Vec<Vec<i32>>) -> i64 {
        a.iter().zip(&b).tuple_combinations().map(|((l1, r1), (l2, r2))|
             (0..2).map(|k| r1[k].min(r2[k]) - l1[k].max(l2[k])).min().unwrap()
        ).max().map_or(0, |x| (x.max(0) as i64).pow(2))
    }
```

