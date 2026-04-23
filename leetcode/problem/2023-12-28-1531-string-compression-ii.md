---
layout: leetcode-entry
title: "1531. String Compression II"
permalink: "/leetcode/problem/2023-12-28-1531-string-compression-ii/"
leetcode_ui: true
entry_slug: "2023-12-28-1531-string-compression-ii"
---

[1531. String Compression II](https://leetcode.com/problems/string-compression-ii/description/) hard
[blog post](https://leetcode.com/problems/string-compression-ii/solutions/4469888/kotlin-dp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28122023-1531-string-compression?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/ajfT7vaAJGY)
![image.png](/assets/leetcode_daily_images/cd5a8f73.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/453

#### Problem TLDR

Min length of run-length encoded aabcc -> a2bc3 after deleting at most k characters

#### Intuition

Let's consider starting from every position, then we can split the problem: result[i] = some_function(i..j) + result[j].

The hardest part is to find an optimal `j` position.

The *wrong* way: trying to count how many s[j]==s[i], and to keep them, removing all other chars s[j]!=s[i]. This didn't give us the optimal solution for s[i..j], as we forced to keep s[0].

The *correct* way: keeping the most frequent char in s[i..j], removing all other chars.

#### Approach

Spend 1-2.5 hours max on the problem, then steal someone else's solution. Don't feel sorry, it's just a numbers game.

#### Complexity

- Time complexity:
$$O(kn^2)$$

- Space complexity:
$$O(kn)$$

#### Code

```kotlin

  fun getLengthOfOptimalCompression(s: String, k: Int): Int {
    val dp = mutableMapOf<Pair<Int, Int>, Int>()
    fun dfs(i: Int, toRemove: Int): Int =
      if (toRemove < 0) Int.MAX_VALUE / 2
      else if (i >= s.length - toRemove) 0
      else dp.getOrPut(i to toRemove) {
        val freq = IntArray(128)
        var mostFreq = 0
        (i..s.lastIndex).minOf { j ->
          mostFreq = max(mostFreq, ++freq[s[j].toInt()])
          when (mostFreq) {
            0 -> 0
            1 -> 1
            else -> mostFreq.toString().length + 1
          } + dfs(j + 1, toRemove - (j - i + 1 - mostFreq))
        }
      }
    return dfs(0, k)
  }

```

