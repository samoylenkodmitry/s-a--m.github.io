---
layout: leetcode-entry
title: "1662. Check If Two String Arrays are Equivalent"
permalink: "/leetcode/problem/2023-12-01-1662-check-if-two-string-arrays-are-equivalent/"
leetcode_ui: true
entry_slug: "2023-12-01-1662-check-if-two-string-arrays-are-equivalent"
---

[1662. Check If Two String Arrays are Equivalent](https://leetcode.com/problems/check-if-two-string-arrays-are-equivalent/description/) easy
[blog post](https://leetcode.com/problems/check-if-two-string-arrays-are-equivalent/solutions/4348780/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01122023-1662-check-if-two-string?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/ewmNv3766OQ)
![image.png](/assets/leetcode_daily_images/e46de001.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/423

#### Problem TLDR

Two dimensional array equals

#### Intuition

There is a one-liner that takes O(n) memory: `ord1.joinToString("") == word2.joinToString("")`. Let's use two-pointer approach to reduce the memory footprint.

#### Approach

* we can iterate with `for` on a first word, and use the pointer variable for the second

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun arrayStringsAreEqual(word1: Array<String>, word2: Array<String>): Boolean {
    var i = 0
    var ii = 0
    for (w in word1) for (c in w) {
      if (i >= word2.size) return false
      if (c != word2[i][ii]) return false
      ii++
      if (ii >= word2[i].length) {
        i++
        ii = 0
      }
    }

    return i == word2.size
  }

```

