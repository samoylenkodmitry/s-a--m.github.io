---
layout: leetcode-entry
title: "131. Palindrome Partitioning"
permalink: "/leetcode/problem/2024-05-22-131-palindrome-partitioning/"
leetcode_ui: true
entry_slug: "2024-05-22-131-palindrome-partitioning"
---

[131. Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/description/) medium
[blog post](https://leetcode.com/problems/palindrome-partitioning/solutions/5191965/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22052024-131-palindrome-partitioning?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ia5Z2-eu7rY)
![2024-05-22_09-02.webp](/assets/leetcode_daily_images/8ded6af4.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/610

#### Problem TLDR

All palindrome partitions #medium #dfs #dynamic_programming

#### Intuition

The backtracking solution is trivial: do a full Depth-First Search over indices, take substring `start..i` if it is a palindrome, collect at the end. We can also precalculate all palindromes in a `dp[i][j] = s[i] == s[j] && dp[i + 1][j - 1]`

#### Approach

However, there is a clever approach to reuse the existing method signature: we can define Dynamic Programming problem as a subproblem for the `palindrome_substring` + DP(`rest of the string`). Where `+` operation would include current palindrome substring in all the suffix's solutions.

Given the problem size, let's skip the memoization part to save lines of code (weird decision for the interview).

#### Complexity

- Time complexity:
$$O(2^n)$$, the worst case is `aaaaa` all chars the same

- Space complexity:
$$O(2^n)$$

#### Code

```kotlin

    fun partition(s: String): List<List<String>> = buildList {
        for (i in s.indices)
            if ((0..i).all { s[it] == s[i - it] })
                if (i < s.lastIndex)
                    for (next in partition(s.drop(i + 1)))
                        add(listOf(s.take(i + 1)) + next)
                else add(listOf(s))
    }

```
```rust

    pub fn partition(s: String) -> Vec<Vec<String>> {
        let mut res = vec![];
        for i in 0..s.len() {
            if (0..=i).all(|j| s.as_bytes()[j] == s.as_bytes()[i - j]) {
                if i < s.len() - 1 {
                    for next in Self::partition(s[i + 1..].to_string()) {
                        res.push(vec![s[..=i].to_string()].into_iter().chain(next).collect())
                    }
                } else { res.push(vec![s.to_string()]) }
            }
        }; res
    }

```

