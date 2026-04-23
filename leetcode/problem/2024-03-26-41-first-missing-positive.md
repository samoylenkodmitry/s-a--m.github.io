---
layout: leetcode-entry
title: "41. First Missing Positive"
permalink: "/leetcode/problem/2024-03-26-41-first-missing-positive/"
leetcode_ui: true
entry_slug: "2024-03-26-41-first-missing-positive"
---

[41. First Missing Positive](https://leetcode.com/problems/first-missing-positive/description/) hard
[blog post](https://leetcode.com/problems/first-missing-positive/solutions/4926741/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26032024-41-first-missing-positive?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/X6syV_fNCo0)
![2024-03-26_09-20.webp](/assets/leetcode_daily_images/a75ad7b4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/550

#### Problem TLDR

First number `1..` not presented in the array, O(1) space #hard

#### Intuition

Let's observe some examples. The idea is to use the array itself, as there is no restriction to modify it:

```j
  /*
  1 -> 2 -> 3 ...

  0 1 2
  1 2 0
  *      0->1->2->0
  0 1 2

  0 1  2 3
  3 4 -1 1
  *         0 -> 3, 3 -> 1, 1 -> 4
  0 1    3 4
       *     2 -> -1

  7 8 9 11 12  1->

   */
```

We can use the indices of array: every present number must be placed at it's index. As numbers are start from `1`, we didn't care about anything bigger than `nums.size`.

#### Approach

* careful with of-by-one's, `1` must be placed at 0 index and so on.

#### Complexity

- Time complexity:
$$O(n)$$, at most twice if all numbers are present in array

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun firstMissingPositive(nums: IntArray): Int {
    for (i in nums.indices)
      while ((nums[i] - 1) in 0..<nums.size && nums[nums[i] - 1] != nums[i])
        nums[nums[i] - 1] = nums[i].also { nums[i] = nums[nums[i] - 1] }
    return nums.indices.firstOrNull { nums[it] != it + 1 }?.inc() ?: nums.size + 1
  }

```
```rust

  pub fn first_missing_positive(mut nums: Vec<i32>) -> i32 {
    let n = nums.len() as i32;
    for i in 0..nums.len() {
      let mut j = nums[i] - 1;
      while 0 <= j && j < n && nums[j as usize] != j + 1 {
        let next = nums[j as usize] - 1;
        nums[j as usize] = j + 1;
        j = next
      }
    }
    for i in 0..n { if nums[i as usize] != i + 1 { return i + 1 }}
    n + 1
  }

```

