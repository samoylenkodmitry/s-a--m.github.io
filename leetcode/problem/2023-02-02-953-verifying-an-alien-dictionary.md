---
layout: leetcode-entry
title: "953. Verifying an Alien Dictionary"
permalink: "/leetcode/problem/2023-02-02-953-verifying-an-alien-dictionary/"
leetcode_ui: true
entry_slug: "2023-02-02-953-verifying-an-alien-dictionary"
---

[953. Verifying an Alien Dictionary](https://leetcode.com/problems/verifying-an-alien-dictionary/description/) easy

[blog post](https://leetcode.com/problems/verifying-an-alien-dictionary/solutions/3130516/kotlin-translate-and-sort/)

```kotlin
    fun isAlienSorted(words: Array<String>, order: String): Boolean {
        val orderChars = Array<Char>(26) { 'a' }
        for (i in 0..25) orderChars[order[i].toInt() - 'a'.toInt()] = (i + 'a'.toInt()).toChar()
        val arr = Array<String>(words.size) {
            words[it].map { orderChars[it.toInt() - 'a'.toInt()] }.joinToString("")
        }

        val sorted = arr.sorted()
        for (i in 0..arr.lastIndex) if (arr[i] != sorted[i]) return false
        return true
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/106
#### Intuition
For the example `hello` and order `hlabcdefgijkmnopqrstuvwxyz` we must translate like this: `h` -> `a`, `l` -> `b`, `a` -> `c` and so on. Then we can just use `compareTo` to check the order.
#### Approach
Just translate and then sort and compare. (But we can also just scan linearly and compare).
#### Complexity
- Time complexity:
  $$O(n\log_2{n})$$
- Space complexity:
  $$O(n)$$

