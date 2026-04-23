---
layout: leetcode-entry
title: "57. Insert Interval"
permalink: "/leetcode/problem/2024-03-17-57-insert-interval/"
leetcode_ui: true
entry_slug: "2024-03-17-57-insert-interval"
---

[57. Insert Interval](https://leetcode.com/problems/insert-interval/description/) medium
[blog post](https://leetcode.com/problems/insert-interval/solutions/4887370/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17032024-57-insert-interval?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/MgX42eP2g0w)
![2024-03-17_10-49.jpg](/assets/leetcode_daily_images/d4af738a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/541

#### Problem TLDR

Insert interval into a sorted intervals array #medium

#### Intuition

There are several ways to attack the problem:
* use single pointer and iterate once
* count prefix and suffix and the middle part
* same as previous, but use the Binary Search

The shortes code is prefix-suffix solution. But you will need to execute some examples to handle indices correctly.
In the interview situation, it is better to start without the BinarySearch part.

#### Approach

To shorted the code let's use some APIs:
* Kotlin: `asList`, `run`, `binarySearchBy`
* Rust: `binary_search_by_key`, `unwrap_or`, `take`, `chain`, `once`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ for the result

#### Code

```kotlin

  fun insert(intervals: Array<IntArray>, newInterval: IntArray) =
    intervals.asList().run {
      var l = binarySearchBy(newInterval[0]) { it[1] }; if (l < 0) l = -l - 1
      var r = binarySearchBy(newInterval[1] + 1) { it[0] }; if (r < 0) r = -r - 1
      val min = min(newInterval[0], (getOrNull(l) ?: newInterval)[0])
      val max = max(newInterval[1], (getOrNull(r - 1) ?: newInterval)[1])
      (take(l) + listOf(intArrayOf(min, max)) + drop(r)).toTypedArray()
    }

```
```rust

  pub fn insert(intervals: Vec<Vec<i32>>, new_interval: Vec<i32>) -> Vec<Vec<i32>> {
    let l = match intervals.binary_search_by_key(&new_interval[0], |x| x[1]) {
        Ok(pos) => pos, Err(pos) => pos };
    let r = match intervals.binary_search_by_key(&(new_interval[1] + 1), |x| x[0]) {
        Ok(pos) => pos, Err(pos) => pos };
    let min_start = new_interval[0].min(intervals.get(l).unwrap_or(&new_interval)[0]);
    let max_end = new_interval[1].max(intervals.get(r - 1).unwrap_or(&new_interval)[1]);
    intervals.iter().take(l).cloned()
    .chain(std::iter::once(vec![min_start, max_end]))
    .chain(intervals.iter().skip(r).cloned()).collect()
  }

```

