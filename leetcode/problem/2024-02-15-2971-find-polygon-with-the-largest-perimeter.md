---
layout: leetcode-entry
title: "2971. Find Polygon With the Largest Perimeter"
permalink: "/leetcode/problem/2024-02-15-2971-find-polygon-with-the-largest-perimeter/"
leetcode_ui: true
entry_slug: "2024-02-15-2971-find-polygon-with-the-largest-perimeter"
---

[2971. Find Polygon With the Largest Perimeter](https://leetcode.com/problems/find-polygon-with-the-largest-perimeter/description/) medium
[blog post](https://leetcode.com/problems/find-polygon-with-the-largest-perimeter/solutions/4729989/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15022024-2971-find-polygon-with-the?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2R628HpYbIg)
![image.png](/assets/leetcode_daily_images/becb03e0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/506

#### Problem TLDR

The largest subset sum(a[..i]) > a[i + 1] where a is a subset of array.

#### Intuition

First, understand the problem: `[1,12,1,2,5,50,3]` doesn't have a polygon, but `[1,12,1,2,5,23,3]` does. After this, the solution is trivial: take numbers in increasing order, compare with sum and check.

#### Approach

Let's try to use the languages.
* Kotlin: `sorted`, `fold`
* Rust: `sort_unstable`, `iter`, `fold`

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$, `sorted` takes O(n) but can be avoided

#### Code

```kotlin

  fun largestPerimeter(nums: IntArray) = nums
    .sorted()
    .fold(0L to -1L) { (s, r), x ->
      s + x to if (s > x) s + x else r
    }.second

```
```rust

  pub fn largest_perimeter(mut nums: Vec<i32>) -> i64 {
    nums.sort_unstable();
    nums.iter().fold((0, -1), |(s, r), &x|
      (s + x as i64, if s > x as i64 { s + x as i64 } else { r })
    ).1
  }

```

