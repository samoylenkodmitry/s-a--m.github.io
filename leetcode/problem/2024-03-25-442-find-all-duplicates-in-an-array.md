---
layout: leetcode-entry
title: "442. Find All Duplicates in an Array"
permalink: "/leetcode/problem/2024-03-25-442-find-all-duplicates-in-an-array/"
leetcode_ui: true
entry_slug: "2024-03-25-442-find-all-duplicates-in-an-array"
---

[442. Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/description/) medium
[blog post](https://leetcode.com/problems/find-all-duplicates-in-an-array/solutions/4922208/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25032024-442-find-all-duplicates?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/iYARBTm85fg)
![2024-03-25_09-19.webp](/assets/leetcode_daily_images/e7c75cbb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/549

#### Problem TLDR

All duplicate numbers of `1..n` using O(1) memory #medium

#### Intuition

There are no restrictions not to modify the input array, so let's flat all visited numbers with a negative sign:

```j

  // 1 2 3 4 5 6 7 8
  // 4 3 2 7 8 2 3 1
  // *     -
  //   * -
  //   - *
  //       *     -
  //         *     -
  //     -     *       --2
  //   -         *     --3
  // -             *

```

Inputs are all positive, the corner cases of negatives and zeros are handled.

#### Approach

* don't forget to `abs`
* Rust didn't permit to iterate and modify at the same time, use pointers

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun findDuplicates(nums: IntArray) = buildList {
    for (x in nums) {
      if (nums[abs(x) - 1] < 0) add(abs(x))
      nums[abs(x) - 1] *= -1
    }
  }

```
```rust

  pub fn find_duplicates(mut nums: Vec<i32>) -> Vec<i32> {
    let mut res = vec![];
    for j in 0..nums.len() {
      let i = (nums[j].abs() - 1) as usize;
      if nums[i] < 0 { res.push(nums[j].abs()) }
      nums[i] *= -1
    }
    res
  }

```

