---
layout: leetcode-entry
title: "1312. Minimum Insertion Steps to Make a String Palindrome"
permalink: "/leetcode/problem/2023-04-22-1312-minimum-insertion-steps-to-make-a-string-palindrome/"
leetcode_ui: true
entry_slug: "2023-04-22-1312-minimum-insertion-steps-to-make-a-string-palindrome"
---

[1312. Minimum Insertion Steps to Make a String Palindrome](https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/description/) hard

```kotlin

fun minInsertions(s: String): Int {
    // abb -> abba
    // abb*
    //  ab -> aba / bab
    // *ab
    //  bba -> abba
    // *bba
    //   bbbaa -> aabbbaa
    // **bbbaa
    //    bbbcaa -> aacbbbcaa
    // ***bbbcaa
    // leetcode ->  leetcodocteel
    // leetcod***e**
    //      o -> 0
    //      od -> dod / dod -> 1
    //     cod -> codoc / docod -> 2
    //     code -> codedoc / edocode -> 2+1=3
    //    tcod -> tcodoct / doctcod -> 2+1=3
    //    tcode -> tcodedoct / edoctcode -> 3+1=4
    //   etcode = e{tcod}e -> e{tcodoct / doctcod}e -> 3
    //   etcod -> 1+{tcod} -> 1+3=4
    //  eetcod -> docteetcod 4 ?/ eetcodoctee 5
    //  eetcode -> edocteetcode 5 / eetcodedoctee 6 -> e{etcod}e 4 = e{etcodocte}e
    // leetcod -> 1+{eetcod} -> 5
    // leetcode -> 1+{eetcode} 1+4=5
    // aboba
    // a -> 0
    // ab -> 1
    // abo -> min({ab}+1, 1+{bo}) =2
    // abob -> min(1+{bob}, {abo} +1)=1
    // aboba -> min(0 + {bob}, 1+{abob}, 1+{boba}) = 0
    val cache = mutableMapOf<Pair<Int, Int>, Int>()
    fun dfs(from: Int, to: Int): Int {
        if (from > to || from < 0 || to > s.lastIndex) return -1
        if (from == to) return 0
        if (from + 1 == to) return if (s[from] == s[to]) 0 else 1
        return cache.getOrPut(from to to) {
            if (s[from] == s[to]) return@getOrPut dfs(from + 1, to - 1)
            val one = dfs(from + 1, to)
            val two = dfs(from, to - 1)
            when {
                one != -1 && two != -1 -> 1 + minOf(one, two)
                one != -1 -> 1 + one
                two != -1 -> 1 + two
                else -> -1
            }
        }
    }
    return dfs(0, s.lastIndex)
}

```

[blog post](https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/solutions/3442679/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-22042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/188
#### Intuition
Let's add chars one by one. Single char is a palindrome. Two chars are palindrome if they are equal, and not if not: $$
insertions_{ab} =\begin{cases}
0, & \text{if a==b}\\
1
\end{cases}
$$. While adding a new character, we choose the minimum insertions. For example, `aboba`:

```

// aboba
// a -> 0
// ab -> 1
// abo -> min({ab}+1, 1+{bo}) =2
// abob -> min(1+{bob}, {abo} +1)=1
// aboba -> min(0 + {bob}, 1+{abob}, 1+{boba}) = 0

```

So, the DP equation is the following $$dp_{i,j} = min(0 + dp_{i+1, j-1}, 1 + dp_{i+1, j}, 1 + dp_{i, j-1}$$, where DP - is the minimum number of insertions.
#### Approach
Just DFS and cache.
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

