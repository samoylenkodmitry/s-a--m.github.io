---
layout: leetcode-entry
title: "3227. Vowels Game in a String"
permalink: "/leetcode/problem/2025-09-12-3227-vowels-game-in-a-string/"
leetcode_ui: true
entry_slug: "2025-09-12-3227-vowels-game-in-a-string"
---

[3227. Vowels Game in a String](https://leetcode.com/problems/vowels-game-in-a-string/description) medium
[blog post](https://leetcode.com/problems/vowels-game-in-a-string/solutions/7181814/kotlin-rust-by-samoylenkodmitry-dyno/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12092025-3227-vowels-game-in-a-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/IX_QFkqmewk)

![1.webp](/assets/leetcode_daily_images/082eeffa.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1110

#### Problem TLDR

Can Alice win Bob both removeing odd-even vowel count substring optimally #medium

#### Intuition

Naive dp: at every position `i` check every position `j=i..n` if the tail is loosing.
Optimization trick: check in reverse `j=n..i` to end game faster

#### Approach

* the final solution is that Alice always wins: odd on the first move, even - on the third

#### Complexity

- Time complexity:
$$O(n^2)$$, or O(n)

- Space complexity:
$$O(n)$$, or O(1)

#### Code

```kotlin

// 16ms
    fun doesAliceWin(s: String) =
        "[aeiou]".toRegex() in s

```
```kotlin

// 33ms
    fun doesAliceWin(s: String): Boolean {
        val f = IntArray(s.length)
        for ((i, c) in s.withIndex()) f[i] = f[max(0, i-1)] + if (c in "aeiou") 1 else 0
        val dp = HashMap<Pair<Int, Boolean>, Boolean>()
        fun dfs(i: Int, odd: Boolean): Boolean =
        i < s.length && dp.getOrPut(i to odd) {
            for (j in s.length-1 downTo i) {
                var cnt = f[j] - (if (i > 0) f[i-1] else 0)
                if (odd == (cnt % 2 > 0))
                if (!dfs(j+1, !odd)) return@getOrPut true
            }
            false
        }
        return dfs(0, true)
    }

```
```rust

// 3ms
    pub fn does_alice_win(s: String) -> bool {
        s.chars().any(|c| "aeiou".contains(c))
    }

```

