---
layout: leetcode-entry
title: "17. Letter Combinations of a Phone Number"
permalink: "/leetcode/problem/2023-08-03-17-letter-combinations-of-a-phone-number/"
leetcode_ui: true
entry_slug: "2023-08-03-17-letter-combinations-of-a-phone-number"
---

[17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/) medium
[blog post](https://leetcode.com/problems/letter-combinations-of-a-phone-number/solutions/3855945/kotlin-dfs-backtracking/)
[substack](https://dmitriisamoilenko.substack.com/p/03082023-17-letter-combinations-of?sd=pf)
![image.png](/assets/leetcode_daily_images/5507fd7c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/297

#### Problem TLDR

Possible words from phone keyboard

#### Intuition

Just a naive DFS and Backtraking will solve the problem, as the number is short

#### Approach

* pay attention to keys in keyboard, some have size of 4

#### Complexity

- Time complexity:
$$O(n4^n)$$, recursion depth is `n`, each time we iterate over '3' or '4' letters, for example:

```
12 ->
abc def
a   d
a    e
a     f
 b  d
 b   e
 b    f
  c d
  c  e
  c   f
```
Each new number multiply previous count by `3` or `4`. The final `joinToString` gives another `n` multiplier.

- Space complexity:
$$O(4^n)$$

#### Code

```kotlin

    fun letterCombinations(digits: String): List<String> = mutableListOf<String>().apply {
      val abc = arrayOf("abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz")
      val list = Stack<Char>()
      fun dfs(pos: Int) {
        if (list.size == digits.length) {
          if (list.isNotEmpty()) add(list.joinToString(""))
        } else abc[digits[pos].toInt() - '2'.toInt()].forEach {
          list.push(it)
          dfs(pos + 1)
          list.pop()
        }
      }
      dfs(0)
    }

```

