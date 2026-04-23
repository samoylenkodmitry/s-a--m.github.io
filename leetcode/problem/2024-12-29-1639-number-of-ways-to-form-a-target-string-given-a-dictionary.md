---
layout: leetcode-entry
title: "1639. Number of Ways to Form a Target String Given a Dictionary"
permalink: "/leetcode/problem/2024-12-29-1639-number-of-ways-to-form-a-target-string-given-a-dictionary/"
leetcode_ui: true
entry_slug: "2024-12-29-1639-number-of-ways-to-form-a-target-string-given-a-dictionary"
---

[1639. Number of Ways to Form a Target String Given a Dictionary](https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/description/) hard
[blog post](https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/solutions/6200889/kotlin-rust-by-samoylenkodmitry-sh2u/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29122024-1639-number-of-ways-to-form?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/1708IKl5qm4)
[deep-dive](https://notebooklm.google.com/notebook/2e187a30-340a-4a8c-97dd-c03deea8383c/audio)
![1.webp](/assets/leetcode_daily_images/6da92d41.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/847

#### Problem TLDR

Ways to make target with increasing positions in words #hard #dynamic_programming

#### Intuition

Let's observe an example at different angles:

```j

    // acca bbbb caca     aba
    // a    .b   ...a
    // a    ..b  ...a
    // a..a .b   ....
    // a..a ..b. ....
    // ...a ..b. .a
    //      ..b  .a.a

    // 0123
    //    a
    // aa a
    // bbbb
    // ccc
    //   c

    //  0 1 2    2 1 0    1 2 0    0 2 1
    // [a,b,c]->[a,b,c]->[b,c,c]->[a,a,b]
    // aba
    //  *          *               *
    //  *          *                 *
    //  *                 *        *
    //  *                 *          *
    //           *        *        *
    //           *        *          *

```

Each position `i` in `words[..]` have a set of chars `words[..][i]`. We can use a full Depth-First Search and `take` or `drop` the current position. To count total ways, we should multiply by count of the taken chars at position. Result can be safely cached by `(i, target_pos)` key.

#### Approach

* we can rewrite top-down DFS + memo into iterative bottom-up DP
* as we only depend on the next (or previous) positions, we can collapse 2D dp into 1D
* some other small optimizations possible, iterate forward for cache-friendliness

#### Complexity

- Time complexity:
$$O(wt)$$

- Space complexity:
$$O(t)$$

#### Code

```kotlin

    fun numWays(words: Array<String>, target: String): Int {
        val fr = Array(words[0].length) { IntArray(26) }
        for (w in words) for (i in w.indices) fr[i][w[i] - 'a']++
        val dp = Array(fr.size + 1) { LongArray(target.length + 1) { -1L }}
        fun dfs(posF: Int, posT: Int): Long = dp[posF][posT].takeIf { it >= 0 } ?: {
            if (posT == target.length) 1L else if (posF == fr.size) 0L else {
                val notTake = dfs(posF + 1, posT)
                val curr = fr[posF][target[posT] - 'a'].toLong()
                val take = if (curr > 0) curr * dfs(posF + 1, posT + 1) else 0
                (take + notTake) % 1_000_000_007L
            }}().also { dp[posF][posT] = it }
        return dfs(0, 0).toInt()
    }

```
```rust

    pub fn num_ways(words: Vec<String>, target: String) -> i32 {
        let mut dp = vec![0; target.len() + 1]; dp[target.len()] = 1;
        let M = 1_000_000_007i64; let target: Vec<_> = target.bytes().rev().collect();
        for posF in 0..words[0].len() {
            let mut fr = vec![0; 26];
            for w in &words { fr[(w.as_bytes()[posF] - b'a') as usize] += 1 }
            for (posT, t) in target.iter().enumerate() {
                dp[posT] += fr[(t - b'a') as usize] * dp[posT + 1] % M
            }
        }; (dp[0] % M) as i32
    }

```

```c++

    int numWays(vector<string>& words, string target) {
        long d[10001] = { 1 }, M = 1e9 + 7;
        for (int i = 0; i < words[0].size(); ++i) {
            int f[26] = {}; for (auto &w: words) ++f[w[i] - 97];
            for (int j = min(i + 1, (int) target.size()); j; --j)
                d[j] += f[target[j - 1] - 97] * d[j - 1] % M;
        } return d[target.size()] % M;
    }

```

