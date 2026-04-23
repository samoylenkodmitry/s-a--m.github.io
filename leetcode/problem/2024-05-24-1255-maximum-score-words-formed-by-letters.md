---
layout: leetcode-entry
title: "1255. Maximum Score Words Formed by Letters"
permalink: "/leetcode/problem/2024-05-24-1255-maximum-score-words-formed-by-letters/"
leetcode_ui: true
entry_slug: "2024-05-24-1255-maximum-score-words-formed-by-letters"
---

[1255. Maximum Score Words Formed by Letters](https://leetcode.com/problems/maximum-score-words-formed-by-letters/description/) hard
[blog post](https://leetcode.com/problems/maximum-score-words-formed-by-letters/solutions/5200230/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24052024-1255-maximum-score-words?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/V96s_V9OXLM)
![2024-05-24_08-45.webp](/assets/leetcode_daily_images/55d86d2a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/612

#### Problem TLDR

Max score of words subset from letters #hard #backtracking #dfs

#### Intuition

This is just a mechanical backtracking problem: do a full Depth-First search over all subsets of words, and count max score.

We can precompute some things beforehead.

#### Approach

* in Kotlin there is a `.code` field, use it
* in Rust: use `[0; 26]` type, it is fast, also use slices, they are cheap and reduce code size

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun maxScoreWords(words: Array<String>, letters: CharArray, score: IntArray): Int {
        val f = IntArray(26); for (c in letters) f[c.code - 'a'.code]++
        val wf = words.map { IntArray(26).apply {
                for (c in it) this[c.code - 'a'.code]++ }}
        val ws = words.map { it.sumOf { score[it.code - 'a'.code] }}
        fun dfs(i: Int): Int = if (i < wf.size) max(dfs(i + 1),
            if ((0..25).all { wf[i][it] <= f[it] }) {
                for (j in 0..25) f[j] -= wf[i][j]
                ws[i] + dfs(i + 1).also { for (j in 0..25) f[j] += wf[i][j] }
            } else 0) else 0
        return dfs(0)
    }

```
```rust

    pub fn max_score_words(words: Vec<String>, letters: Vec<char>, score: Vec<i32>) -> i32 {
        let (mut f, mut wf, mut ws) = ([0; 26], vec![[0; 26]; words.len()], vec![0; words.len()]);
        for &c in letters.iter() { f[(c as u8 - b'a') as usize] += 1 }
        for (i, w) in words.iter().enumerate() {
            for b in w.bytes() { wf[i][(b - b'a') as usize] += 1; ws[i] += score[(b - b'a') as usize] }
        }
        fn dfs(f: &mut [i32; 26], ws: &[i32], wf: &[[i32; 26]]) -> i32 {
            if wf.len() > 0 { dfs(f, &ws[1..], &wf[1..]).max(
                if (0..25).all(|i| wf[0][i] <= f[i]) {
                    for i in 0..25 { f[i] -= wf[0][i] }
                    let next = ws[0] + dfs(f, &ws[1..], &wf[1..]);
                    for i in 0..25 { f[i] += wf[0][i] }; next
                } else { 0 }) } else { 0 }
        } dfs(&mut f, &ws, &wf)
    }

```

