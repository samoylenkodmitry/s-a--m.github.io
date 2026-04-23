---
layout: leetcode-entry
title: "977. Squares of a Sorted Array"
permalink: "/leetcode/problem/2024-03-02-977-squares-of-a-sorted-array/"
leetcode_ui: true
entry_slug: "2024-03-02-977-squares-of-a-sorted-array"
---

[977. Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/description/) easy
[blog post](https://leetcode.com/problems/squares-of-a-sorted-array/solutions/4808833/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02032024-977-squares-of-a-sorted?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ytGHSEDwtgs)
![image.png](/assets/leetcode_daily_images/2a47234d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/526

#### Problem TLDR

Sorted squares.

#### Intuition

We can build the result bottom up or top down. Either way, we need two pointers: for the negative and for the positive.

#### Approach

Can we made it shorter?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun sortedSquares(nums: IntArray): IntArray {
    var i = 0; var j = nums.lastIndex;
    return IntArray(nums.size) {
      (if (abs(nums[i]) > abs(nums[j]))
        nums[i++] else nums[j--]).let { it * it }
    }.apply { reverse() }
  }

```
```rust

  pub fn sorted_squares(nums: Vec<i32>) -> Vec<i32> {
    let (mut i, mut j) = (0, nums.len() - 1);
    let mut v: Vec<_> = (0..=j).map(|_|
      if nums[i].abs() > nums[j].abs() {
        i += 1; nums[i - 1] * nums[i - 1]
      } else { j -= 1; nums[j + 1] * nums[j + 1] })
      .collect(); v.reverse(); v
  }

```

