---
layout: leetcode-entry
title: "1980. Find Unique Binary String"
permalink: "/leetcode/problem/2023-11-16-1980-find-unique-binary-string/"
leetcode_ui: true
entry_slug: "2023-11-16-1980-find-unique-binary-string"
---

[1980. Find Unique Binary String](https://leetcode.com/problems/find-unique-binary-string/description/) medium
[blog post](https://leetcode.com/problems/find-unique-binary-string/solutions/4293360/kotlin-sort/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16112023-1980-find-unique-binary?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/6b065f8f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/406

#### Problem TLDR

First absent number in a binary string array

#### Intuition

The naive solution would be searching in all the numbers `0..2^n`. However, if we convert strings to ints and sort them, we can do a linear scan to detect first absent.

#### Approach

* use padStart to convert back

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun findDifferentBinaryString(nums: Array<String>): String {
      var next = 0
      for (x in nums.sorted()) {
        if (x.toInt(2) > next) break
        next++
      }
      return next.toString(2).padStart(nums[0].length, '0')
    }

```

