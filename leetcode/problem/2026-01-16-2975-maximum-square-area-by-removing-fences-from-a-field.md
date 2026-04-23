---
layout: leetcode-entry
title: "2975. Maximum Square Area by Removing Fences From a Field"
permalink: "/leetcode/problem/2026-01-16-2975-maximum-square-area-by-removing-fences-from-a-field/"
leetcode_ui: true
entry_slug: "2026-01-16-2975-maximum-square-area-by-removing-fences-from-a-field"
---

[2975. Maximum Square Area by Removing Fences From a Field](https://leetcode.com/problems/maximum-square-area-by-removing-fences-from-a-field/description/) medium
[blog post](https://leetcode.com/problems/maximum-square-area-by-removing-fences-from-a-field/solutions/7499118/kotlin-rust-by-samoylenkodmitry-3kdt/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16012026-2975-maximum-square-area?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/wIicj0TOIEw)

![85f3cd69-628c-42a4-9194-065e536e30a8 (1).webp](/assets/leetcode_daily_images/29836401.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1239

#### Problem TLDR

Max square after remove vertical or horizontal bars #medium

#### Intuition

```j
    // possible horizontal widths
    // intersect
    // possible vertical heights
    //
    // 1D problem: collect possible gaps
    //
    // |*****|**|*|
    //    5    2 1
    // 1,2,5, 1+2, 2+5, 1+2+5
    //             |+1 -- all prev + x
    // 1+1,(1+2)+1,(1+2+5)+1
    // max size is 600, so O(n^2) should be acceptable
```

Solve 1D problem. Find all possible gaps. Intersect. Pick max.

#### Approach

* to find all gaps just use 2d for loop

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
// 648ms
    fun maximizeSquareArea(m: Int, n: Int, h: IntArray, v: IntArray) = listOf(h+1+m,v+1+n)
        .map { buildSet { for (a in it) for (b in it) if (a>b) add(a-b) }}
        .let {(a,b)->a.intersect(b).maxOrNull()}?.let {1L*it*it%1000000007} ?: -1
```
```rust
// 355ms
    pub fn maximize_square_area(m: i32, n: i32, mut h: Vec<i32>, mut v: Vec<i32>) -> i32 {
        let [a,b] = [(h,m),(v,n)].map(|(mut x,y)| {
            let mut s = HashSet::new(); x.extend([1,y]);
            for a in &x { for b in &x { if a > b { s.insert(a-b); }}}; s });
        a.intersection(&b).max().map_or(-1, |&x| ((x as i64*x as i64)%1000000007) as i32)
    }
```

