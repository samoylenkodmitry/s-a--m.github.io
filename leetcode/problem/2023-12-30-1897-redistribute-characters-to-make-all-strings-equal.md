---
layout: leetcode-entry
title: "1897. Redistribute Characters to Make All Strings Equal"
permalink: "/leetcode/problem/2023-12-30-1897-redistribute-characters-to-make-all-strings-equal/"
leetcode_ui: true
entry_slug: "2023-12-30-1897-redistribute-characters-to-make-all-strings-equal"
---

[1897. Redistribute Characters to Make All Strings Equal](https://leetcode.com/problems/redistribute-characters-to-make-all-strings-equal/description/) easy
[blog post](https://leetcode.com/problems/redistribute-characters-to-make-all-strings-equal/solutions/4477383/kotlin-frequency/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30122023-1897-redistribute-characters?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/ltXmpMv4wHo)
![image.png](/assets/leetcode_daily_images/843a92cf.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/455

#### Problem TLDR

Is it possible to split all the words[] characters into words.size groups.

#### Intuition

To understand the problem, consider example: `a abc abbcc` -> `[abc] [abc] [abc]`. We know the result words count, and we know the count of each kind of character. So, just make sure, every character's count can be separated into `words.size` groups.

#### Approach

* to better understand the problem, consider adding more examples
* there can be more than one repeating character in group, `[aabc] [aabc] [aabc]`

#### Complexity

- Time complexity:
$$O(nw)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun makeEqual(words: Array<String>) =
    IntArray(26).apply {
      for (w in words) for (c in w) this[c.toInt() - 'a'.toInt()]++
    }.all { it % words.size == 0 }

```

