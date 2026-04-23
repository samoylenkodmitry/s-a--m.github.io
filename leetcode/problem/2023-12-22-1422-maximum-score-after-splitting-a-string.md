---
layout: leetcode-entry
title: "1422. Maximum Score After Splitting a String"
permalink: "/leetcode/problem/2023-12-22-1422-maximum-score-after-splitting-a-string/"
leetcode_ui: true
entry_slug: "2023-12-22-1422-maximum-score-after-splitting-a-string"
---

[1422. Maximum Score After Splitting a String](https://leetcode.com/problems/maximum-score-after-splitting-a-string/description/) easy
[blog post](https://leetcode.com/problems/maximum-score-after-splitting-a-string/solutions/4440027/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22122023-1422-maximum-score-after?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
![image.png](/assets/leetcode_daily_images/401fe726.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/447

#### Problem TLDR

Max left_zeros + right_ones in 01-array

#### Intuition

We can count `ones` and then scan from the beginning modifying the `ones` and `zeros` counts. After some retrospect, we can do this with `score` variable.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$, dropLast(1) creates the second list, but we can just use pointers or `asSequence`

#### Code

```kotlin

    fun maxScore(s: String): Int {
      var score = s.count { it == '1' }
      return s.dropLast(1).maxOf {
        if (it == '0') ++score else --score
      }
    }

```

