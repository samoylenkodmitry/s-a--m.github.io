---
layout: leetcode-entry
title: "459. Repeated Substring Pattern"
permalink: "/leetcode/problem/2023-08-21-459-repeated-substring-pattern/"
leetcode_ui: true
entry_slug: "2023-08-21-459-repeated-substring-pattern"
---

[459. Repeated Substring Pattern](https://leetcode.com/problems/repeated-substring-pattern/description/) easy
[blog post](https://leetcode.com/problems/repeated-substring-pattern/solutions/3939069/kotlin-rolling-hash/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21082023-459-repeated-substring-pattern?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/e9c1e380.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/315

#### Intuition

Consider example, `abc abc abc`. Doing shift left `3` times we get the same string:

```
abcabcabc - original
bcabcabca - shift left by 1
cabcabcab - shift left by 1
abcabcabc - shift left by 1
```

Now, there is a technique called Rolling Hash: let's calculate the hash like this: `hash = x + 31 * hash`. After full string hash calculated, we start doing shifts:

```
    // abcd
    // a
    // 32^0 * b + 32^1 * a
    // 32^0 * c + 32^1 * b + 32^2 * a
    // 32^0 * d + 32^1 * c + 32^2 * b + 32^3 * a
    // bcda
    // 32^0 * a + 32^1 * d + 32^2 * c + 32^3 * b = 32*(abcd-32^3a) +a=32abcd-(32^4-1)a
```
Observing this math equation, next rolling hash is `shiftHash = 31 *  shiftHash - 31^len + c`

#### Approach

* careful to not shift by whole length

#### Complexity

- Time complexity:
$$O(n)$$, at most 2 full scans, and hashing gives O(1) time

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun repeatedSubstringPattern(s: String): Boolean {
      var hash = 0L
      for (c in s) hash = c.toInt() + 31L * hash
      var pow = 1L
      repeat(s.length) { pow *= 31L }
      pow--
      var shiftHash = hash
      return (0 until s.lastIndex).any { i ->
        shiftHash = 31L * shiftHash - pow * s[i].toInt()
        shiftHash == hash &&
          s == s.substring(0, i + 1).let { it.repeat(s.length / it.length) }
      }
    }

```

