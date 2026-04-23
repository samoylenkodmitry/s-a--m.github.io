---
layout: leetcode-entry
title: "2433. Find The Original Array of Prefix Xor"
permalink: "/leetcode/problem/2023-10-31-2433-find-the-original-array-of-prefix-xor/"
leetcode_ui: true
entry_slug: "2023-10-31-2433-find-the-original-array-of-prefix-xor"
---

[2433. Find The Original Array of Prefix Xor](https://leetcode.com/problems/find-the-original-array-of-prefix-xor/description/) medium
[blog post](https://leetcode.com/problems/find-the-original-array-of-prefix-xor/solutions/4229075/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31102023-2433-find-the-original-array?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/12e7f0b4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/387

#### Problem TLDR

Reverse `xor` operation

#### Intuition

Let's observe how `xor` works:

```kotlin
    // 010 2
    // 101 5
    // 111 7
    // 5 xor 7 = 2
    // 101 xor 111 = 010
    // 5 xor 2 = 101 xor 010 = 111
```
We can reverse the `xor` operation by applying it again: `a ^ b = c`, then `a ^ c = b`

#### Approach

There are several ways to write this:

1. by using `mapIndexed`
2. by in-place iteration
3. by creating a new array

Let's use Kotlin's array constructor lambda and `getOrElse`.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun findArray(pref: IntArray) = IntArray(pref.size) {
      pref[it] xor pref.getOrElse(it - 1) { 0 }
    }

```

