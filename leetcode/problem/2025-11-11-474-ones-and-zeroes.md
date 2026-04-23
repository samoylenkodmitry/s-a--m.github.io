---
layout: leetcode-entry
title: "474. Ones and Zeroes"
permalink: "/leetcode/problem/2025-11-11-474-ones-and-zeroes/"
leetcode_ui: true
entry_slug: "2025-11-11-474-ones-and-zeroes"
---

[474. Ones and Zeroes](https://leetcode.com/problems/ones-and-zeroes/description) medium
[blog post](https://leetcode.com/problems/ones-and-zeroes/solutions/7341279/kotlin-rust-by-samoylenkodmitry-egjh/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11112025-474-ones-and-zeroes?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NADW0Fr6mI4)

![1f7cb25c-39cc-4cf6-ae33-8e0345abf2f4 (1).webp](/assets/leetcode_daily_images/9d1aa252.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1170

#### Problem TLDR

Longest subsequence (ones,zeros) at most (m,n) #medium #dp

#### Intuition

At each string do a choice take it or not. Cache by (i, zeros, ones).

#### Approach

* bottom up intuition: "update [zeros x ones] matrix if we take the current string"

#### Complexity

- Time complexity:
$$O(nl^2)$$

- Space complexity:
$$O(nl^2)$$

#### Code

```kotlin
// 179ms
    fun findMaxForm(s: Array<String>, m: Int, n: Int): Int {
        val os = s.map { it.count { it == '1' }}; val dp = HashMap<Int, Int>();
        fun dfs(i: Int, no: Int, nz: Int): Int = dp.getOrPut(i*10000+no*100+nz) {
            if (no > n || nz > m) return Int.MIN_VALUE / 2; if (i == s.size) return 0
            max(1 + dfs(i + 1, no + os[i], nz + s[i].length - os[i]), dfs(i + 1, no, nz))
        }
        return dfs(0, 0, 0)
    }
```
```rust
// 17ms
    pub fn find_max_form(s: Vec<String>, m: i32, n: i32) -> i32 {
        let (m,n)=(m as usize, n as usize); let mut dp = vec![vec![0; n+1];m+1];
        for s in s {
            let o = s.matches('1').count(); let z = s.len() - o;
            for i in (z..=m).rev() { for j in (o..=n).rev() {
                dp[i][j] = dp[i][j].max(dp[i - z][j - o] + 1) }}
        }; dp[m][n]
    }
```

