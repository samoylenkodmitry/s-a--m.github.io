---
layout: leetcode-entry
title: "2962. Count Subarrays Where Max Element Appears at Least K Times"
permalink: "/leetcode/problem/2024-03-29-2962-count-subarrays-where-max-element-appears-at-least-k-times/"
leetcode_ui: true
entry_slug: "2024-03-29-2962-count-subarrays-where-max-element-appears-at-least-k-times"
---

[2962. Count Subarrays Where Max Element Appears at Least K Times](https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times/description/) medium
[blog post](https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times/solutions/4940899/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29032024-2962-count-subarrays-where?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/d0Je22SXmlE)
![2024-03-29_09-26.webp](/assets/leetcode_daily_images/50472f78.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/553

#### Problem TLDR

Count subarrays with at least `k` array max in #medium

#### Intuition

Let's observe an example `1 3 3`:

```j
    // inverse the problem
    // [1], [3], [3], [1 3], [1 3 3], [3 3] // 6
    // 1 3 3     ck  c
    // j .
    // i .           1
    //   i        1  3
    //     i      2
    //   j
    //     j      1  4
    //                          6-4=2
```
The problem is more simple if we invert it: count subarrays with less than `k` maximums. Then it is just a two-pointer problem: increase by one, then shrink until condition `< k` met.

Another way, is to solve problem at face: left border is the count we need - all subarrays before our `j..i` will have `k` max elements if `j..i` have them.

#### Approach

Let's implement both.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun countSubarrays(nums: IntArray, k: Int): Long {
    val n = nums.size.toLong()
    val m = nums.max(); var ck = 0; var j = 0
    return n * (n + 1) / 2 + nums.indices.sumOf { i ->
      if (nums[i] == m) ck++
      while (ck >= k) if (nums[j++] == m) ck--
      -(i - j + 1).toLong()
    }
  }

```
```rust

  pub fn count_subarrays(nums: Vec<i32>, k: i32) -> i64 {
    let (mut j, mut curr, m) = (0, 0, *nums.iter().max().unwrap());
    (0..nums.len()).map(|i| {
      if nums[i] == m { curr += 1 }
      while curr >= k { if nums[j] == m { curr -= 1 }; j += 1 }
      j as i64
    }).sum()
  }

```

