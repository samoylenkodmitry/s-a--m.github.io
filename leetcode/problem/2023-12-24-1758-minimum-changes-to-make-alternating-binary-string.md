---
layout: leetcode-entry
title: "1758. Minimum Changes To Make Alternating Binary String"
permalink: "/leetcode/problem/2023-12-24-1758-minimum-changes-to-make-alternating-binary-string/"
leetcode_ui: true
entry_slug: "2023-12-24-1758-minimum-changes-to-make-alternating-binary-string"
---

[1758. Minimum Changes To Make Alternating Binary String](https://leetcode.com/problems/minimum-changes-to-make-alternating-binary-string/description/) easy
[blog post](https://leetcode.com/problems/minimum-changes-to-make-alternating-binary-string/solutions/4450527/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24122023-1758-minimum-changes-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/LycsaL5IeTk)
![image.png](/assets/leetcode_daily_images/3f12a28b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/449

#### TLDR

Minimum operations to make `01`-string with no two adjacent equal

#### Intuition

There are only two possible final variations - odd zeros even ones or even zeros odd ones. We can count how many positions to changes for each of them, then return smallest counter.

#### Approach

In a stressfull situation better to just use 4 counters: oddOnes, evenOnes, oddZeros, evenZeros. Then do something with them.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun minOperations(s: String): Int {
    var oddOnesEvenZeros = 0
    var oddZerosEvenOnes = 0
    for (i in s.indices) when {
      s[i] == '0' && i % 2 == 0 -> oddZerosEvenOnes++
      s[i] == '0' && i % 2 != 0 -> oddOnesEvenZeros++
      s[i] == '1' && i % 2 == 0 -> oddOnesEvenZeros++
      s[i] == '1' && i % 2 != 0 -> oddZerosEvenOnes++
    }
    return min(oddOnesEvenZeros, oddZerosEvenOnes)
  }

```

