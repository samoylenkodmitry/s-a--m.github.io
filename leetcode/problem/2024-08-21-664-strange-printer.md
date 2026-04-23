---
layout: leetcode-entry
title: "664. Strange Printer"
permalink: "/leetcode/problem/2024-08-21-664-strange-printer/"
leetcode_ui: true
entry_slug: "2024-08-21-664-strange-printer"
---

[664. Strange Printer](https://leetcode.com/problems/strange-printer/solutions/) hard
[blog post](https://leetcode.com/problems/strange-printer/solutions/5667966/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21082024-664-strange-printer?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/g9BtEFTis3Q)
![1.webp](/assets/leetcode_daily_images/556cb29f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/709

#### Problem TLDR

Minimum continuous replacements to make a string #hard #dynamic_programming

# Intuition

Last time I solved it fine (1 year ago, https://t.me/leetcode_daily_unstoppable/291), this time, however, I was stuck with the corner cases, ultimately failing to solve it in 1.5 hours.

The not working idea was to consider the case of painting the [i..j] substring when its endings are equal s[i] == s[j], and choose between repainting entire thing or just appending one symbol. This also has to consider the `background` color already painted, so it is dp[i][j][b]:

```j

    // abcabc
    // aaaaaa
    //  bb
    //   c
    //     bb
    //      c

    // abcdcba     cdc + ab..ba, cdc = d + c..c,  cd = d + c.. or c + ..d
    //             cdcba = c + ..dcba or cdcb + ..a
    // cdc = 1 + min(cd + c, c + dc)

```
The `if` tree grown too much, and some cases were failing, and I still think I missing some cases or idea is just completely wrong.

The working idea: try all possible splits to paint and choose the minimum.

#### Approach

Let's implement both recursive and bottom-up iterative solutions.

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun strangePrinter(s: String): Int {
        val dp = mutableMapOf<Pair<Int, Int>, Int>()
        fun dfs(i: Int, j: Int): Int =
            if (i == j) 1 else if (i > j) 0
            else if (i == j - 1) { if (s[i] == s[j]) 1 else 2 }
            else dp.getOrPut(i to j) {
                if (s[i] == s[i + 1]) dfs(i + 1, j)
                else if (s[j] == s[j - 1]) dfs(i, j - 1)
                else (i..j - 1).map { dfs(i, it) + dfs(it + 1, j) }.min() -
                    if (s[i] == s[j]) 1 else 0
            }
        return dfs(0, s.lastIndex)
    }

```
```rust

    pub fn strange_printer(s: String) -> i32 {
        let n = s.len(); let mut dp = vec![vec![-1; n]; n];
        let s = s.as_bytes();
        for (j, &b) in s.iter().enumerate() {
            for i in (0..=j).rev() {
                dp[i][j] = if j - i <= 1 { if s[i] == b { 1 } else { 2 }}
                else if s[i] == b { dp[i + 1][j] }
                else {
                    (i..j).map(|k| dp[i][k] + dp[k + 1][j] ).min().unwrap() -
                    if s[i] == b { 1 } else { 0 }
                }}}
        dp[0][n - 1]
    }

```

