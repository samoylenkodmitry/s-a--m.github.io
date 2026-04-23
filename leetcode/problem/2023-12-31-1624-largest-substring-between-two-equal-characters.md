---
layout: leetcode-entry
title: "1624. Largest Substring Between Two Equal Characters"
permalink: "/leetcode/problem/2023-12-31-1624-largest-substring-between-two-equal-characters/"
leetcode_ui: true
entry_slug: "2023-12-31-1624-largest-substring-between-two-equal-characters"
---

[1624. Largest Substring Between Two Equal Characters](https://leetcode.com/problems/largest-substring-between-two-equal-characters/description/) easy
[blog post](https://leetcode.com/problems/largest-substring-between-two-equal-characters/solutions/4482196/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31122023-1624-largest-substring-between?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/BF4M70PncfE)
![image.png](/assets/leetcode_daily_images/fa0ffef7.webp)
https://youtu.be/BF4M70PncfE

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/456

#### Problem TLDR

Max distance between same chars in string.

#### Intuition

We must remember the first occurrence position of each kind of character.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun maxLengthBetweenEqualCharacters(s: String) =
    with(mutableMapOf<Char, Int>()) {
      s.indices.maxOf { it - 1 - getOrPut(s[it]) { it } }
    }

```

