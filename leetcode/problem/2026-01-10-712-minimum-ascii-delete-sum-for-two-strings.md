---
layout: leetcode-entry
title: "712. Minimum ASCII Delete Sum for Two Strings"
permalink: "/leetcode/problem/2026-01-10-712-minimum-ascii-delete-sum-for-two-strings/"
leetcode_ui: true
entry_slug: "2026-01-10-712-minimum-ascii-delete-sum-for-two-strings"
---

[712. Minimum ASCII Delete Sum for Two Strings](https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/description) medium
[blog post](https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/solutions/7483288/kotlin-rust-by-samoylenkodmitry-gmd8/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10012026-712-minimum-ascii-delete?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Goqrb3lwg5o)

![2beff096-6071-468f-934f-c4137d076ed3 (1).webp](/assets/leetcode_daily_images/833ae522.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1233

#### Problem TLDR

Min removed sum to make strings equal #medium #dp

#### Intuition

One way:
DP[i][j] is the removed sum for suffix a[i..],b[j..], take a[i] or b[j] when a[i]!=b[j].

Second way:
DP[i][j] is the Longest Common Substring for suffix a[i..],b[j..], take a[i] when a[i]==b[j].

#### Approach

* the DFS top-down is easier to write
* space optimize with i%2 trick: only the last row is needed
* forward write to simplify indices dp[i+1][j+1] = ... a[i],b[j]

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(m)$$

#### Code

```kotlin
// 30ms
    fun minimumDeleteSum(a: String, b: String): Int {
        val d = Array(2) { IntArray(b.length+1) }
        for (i in a.indices) for (j in b.indices) d[(i+1)%2][j+1] =
            if (a[i] == b[j]) a[i].code + d[i%2][j] else max(d[i%2][j+1], d[(i+1)%2][j])
        return a.sumOf{it.code}+b.sumOf{it.code}-2*d[a.length%2][b.length]
    }
```
```rust
// 3ms
    pub fn minimum_delete_sum(a: String, b: String) -> i32 {
        let (a, b, mut d) = (a.as_bytes(), b.as_bytes(), vec![vec![0; b.len() + 1]; 2]);
        for j in 0..b.len() { d[0][j + 1] = d[0][j] + b[j] as i32 }
        for i in 0..a.len() { for j in 0..=b.len() { d[(i + 1) & 1][j] =
            if j < 1 { d[i & 1][0] + a[i] as i32 } else if a[i] == b[j - 1] { d[i & 1][j - 1] }
            else { (a[i] as i32 + d[i & 1][j]).min(b[j - 1] as i32 + d[(i + 1) & 1][j - 1]) }
        }}
        d[a.len() & 1][b.len()]
    }
```

