---
layout: leetcode-entry
title: "452. Minimum Number of Arrows to Burst Balloons"
permalink: "/leetcode/problem/2024-03-18-452-minimum-number-of-arrows-to-burst-balloons/"
leetcode_ui: true
entry_slug: "2024-03-18-452-minimum-number-of-arrows-to-burst-balloons"
---

[452. Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/solutions/4891442/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18032024-452-minimum-number-of-arrows?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/SjkSJIF6Z_g)
![2024-03-18_09-23.jpg](/assets/leetcode_daily_images/0360decf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/542

#### Problem TLDR

Count non-intersecting intervals #medium

#### Intuition

After sorting, we can line-sweep scan the intervals and count non-intersected ones.
The edge case is that the `right` scan border will shrink to the smallest.

```j

 [3,9],[7,12],[3,8],[6,8],[9,10],[2,9],[0,9],[3,9],[0,6],[2,8]
 0..9 0..6 2..9 2..8 3..9 3..8 3..9 6..8 7..12 9..10
    * -  6 -    -    -    -    -    -    |

```

#### Approach

Let's do some codegolf with Kotlin

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$, or O(n) with `sortedBy`

#### Code

```kotlin

  fun findMinArrowShots(points: Array<IntArray>): Int =
    1 + points.sortedBy { it[0] }.let { p -> p.count { (from, to) ->
      (from > p[0][1]).also {
        p[0][1] = min(if (it) to else p[0][1], to) }}}

```
```rust

  pub fn find_min_arrow_shots(mut points: Vec<Vec<i32>>) -> i32 {
    points.sort_unstable_by_key(|p| p[0]);
    let (mut shoots, mut right) = (1, points[0][1]);
    for p in points {
      if p[0] > right { shoots += 1; right = p[1] }
      right = right.min(p[1])
    }; shoots
  }

```

