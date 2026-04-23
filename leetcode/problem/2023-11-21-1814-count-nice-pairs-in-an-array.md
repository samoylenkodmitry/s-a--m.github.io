---
layout: leetcode-entry
title: "1814. Count Nice Pairs in an Array"
permalink: "/leetcode/problem/2023-11-21-1814-count-nice-pairs-in-an-array/"
leetcode_ui: true
entry_slug: "2023-11-21-1814-count-nice-pairs-in-an-array"
---

[1814. Count Nice Pairs in an Array](https://leetcode.com/problems/count-nice-pairs-in-an-array/description/) medium
[blog post](https://leetcode.com/problems/count-nice-pairs-in-an-array/solutions/4312107/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21112023-1814-count-nice-pairs-in?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/42102f4c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/412

#### Problem TLDR

Count pairs `x-rev(x) == y-rev(y)`, where `rev(123) = 321`

#### Intuition

For simplicity, let's redefine the equation, keeping `i` and `j` on a separate parts  $$nums[i] - rev(nums[i]) == nums[j] - rev(nums[j])$$. Now, we can precompute `nums[i] - rev(nums[i])`. The remaining part of an algorithm is how to calculate count of the duplicate numbers in a linear scan.

#### Approach

Let's use a HashMap to count the previous numbers count. Each new number will make a `count` new pairs.

#### Complexity

- Time complexity:
$$O(nlg(n))$$, lg(n) - for the `rev()`

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun countNicePairs(nums: IntArray): Int {
    val counts = HashMap<Int, Int>()
    var sum = 0
    for (x in nums) {
      var n = x
      var rev = 0
      while (n > 0) {
        rev = (n % 10) + rev * 10
        n = n / 10
      }
      val count = counts[x - rev] ?: 0
      sum = (sum + count) % 1_000_000_007
      counts[x - rev] = count + 1
    }
    return sum
  }

```

