---
layout: leetcode-entry
title: "229. Majority Element II"
permalink: "/leetcode/problem/2023-10-05-229-majority-element-ii/"
leetcode_ui: true
entry_slug: "2023-10-05-229-majority-element-ii"
---

[229. Majority Element II](https://leetcode.com/problems/majority-element-ii/description/) medium
[blog post](https://leetcode.com/problems/majority-element-ii/solutions/4131903/kotlin-moore-algo/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/5102023-229-majority-element-ii?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/7358655b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/360

#### Problem TLDR

Elements with frequency > size / 3

#### Intuition

The naive solution, which is to count frequencies, can be this one-liner:

```kotlin
    fun majorityElement(nums: IntArray) = nums
      .groupBy { it }
      .filter { (k, v) -> v.size > nums.size / 3 }
      .map { (k, v) -> k }
```

However, to solve it in O(1) we need to read the `hint`: Moore algo.
One idea is that there are at most only `two` such elements can coexist:
```
    // 111 123 333
    // 1111 1234 4444
    // 11111 12345 55555
```
The second idea is a clever counting of `three` buckets: `first` candidate, `second` candidate and others. We decrease candidates counters if `x` in the `others` bucket, and change candidate if it's counter `0`.

#### Approach

Steal someone's else solution or ask ChatGPT about `Moore` algorithm to find majority element.
* make sure you understand why the resulting elements are majority

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun majorityElement(nums: IntArray): List<Int> {
      var x1 = Int.MIN_VALUE
      var x2 = Int.MIN_VALUE
      var count1 = 0
      var count2 = 0
      for (x in nums) when {
        x != x2 && count1 == 0 -> x1 = x.also { count1 = 1 }
        x != x1 && count2 == 0 -> x2 = x.also { count2 = 1 }
        x == x1 -> count1++
        x == x2 -> count2++
        else -> {
          count1 = maxOf(0, count1 - 1)
          count2 = maxOf(0, count2 - 1)
        }
      }
      return buildList {
        if (nums.count { it == x1 } > nums.size / 3) add(x1)
        if (nums.count { it == x2 } > nums.size / 3) add(x2)
      }
    }

```

