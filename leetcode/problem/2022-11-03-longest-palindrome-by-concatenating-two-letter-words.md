---
layout: leetcode-entry
title: "Longest Palindrome By Concatenating Two Letter Words"
permalink: "/leetcode/problem/2022-11-03-longest-palindrome-by-concatenating-two-letter-words/"
leetcode_ui: true
entry_slug: "2022-11-03-longest-palindrome-by-concatenating-two-letter-words"
---

[https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/](https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/) medium

Solution [kotlin]

```kotlin

fun longestPalindrome(words: Array<String>): Int {
        var singles = 0
        var mirrored = 0
        var uneven = 0
        var unevenSum = 0
        val visited = mutableMapOf<String, Int>()
        words.forEach { w ->  visited[w] = 1 + visited.getOrDefault(w, 0) }
        visited.forEach { w, wCount ->
            if (w[0] == w[1]) {
                if (wCount %2 == 0) {
                    singles += wCount*2
                } else {
                    // a b -> a
                    // a b a -> aba 2a + 1b = 2 + 1
                    // a b a b -> abba 2a + 2b = 2+2
                    // a b a b a -> baaab 3a + 2b = 3+2
                    // a b a b a b -> baaab 3a + 3b = 3+2 (-1)
                    // a b a b a b a -> aabbbaa 4a+3b=4+3
                    // a b a b a b a b -> aabbbbaa 4a+4b=4+4
                    // 5a+4b = 2+5+2
                    // 5a+5b = 2+5+2 (-1)
                    // 1c + 2b + 2a = b a c a b
                    // 1c + 3b + 2a =
                    // 1c + 3b + 4a = 2a + 3b + 2a
                    // 5d + 3a + 3b + 3c = a b c 5d c b a = 11
                    uneven++
                    unevenSum += wCount
                }
            } else {
                val matchingCount = visited[w.reversed()] ?:0
                mirrored += minOf(wCount, matchingCount)*2
            }
        }
        val unevenCount = if (uneven == 0) 0 else 2*(unevenSum - uneven + 1)
        return singles + mirrored + unevenCount
    }

```

Explanation:
This is a counting task, can be solved linearly.
There are 3 cases:
1. First count mirrored elements, "ab" <-> "ba", they all can be included to the result
2. Second count doubled letters "aa", "bb". Notice, that if count is even, they also can be splitted by half and all included.
3. The only edge case is uneven part. The law can be derived by looking at the examples

Speed: O(N), Memory O(N)

