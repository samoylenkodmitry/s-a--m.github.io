---
layout: leetcode-entry
title: "1458. Max Dot Product of Two Subsequences"
permalink: "/leetcode/problem/2026-01-08-1458-max-dot-product-of-two-subsequences/"
leetcode_ui: true
entry_slug: "2026-01-08-1458-max-dot-product-of-two-subsequences"
---

[1458. Max Dot Product of Two Subsequences](https://leetcode.com/problems/max-dot-product-of-two-subsequences/description) hard
[blog post](https://leetcode.com/problems/max-dot-product-of-two-subsequences/solutions/7477454/kotlin-rust-by-samoylenkodmitry-yglc/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08012026-1458-max-dot-product-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2J8y-Uc-Ubc)

![f5ec9834-1013-4da3-bdd2-385fe23dd48d (1).webp](/assets/leetcode_daily_images/4f65276e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1231

#### Problem TLDR

Max product of 2 subsequencies #hard #dp

#### Intuition

The solution is stable for every suffix (a[i..],b[j..]), memoize by the key of (i,j).

#### Approach

* use forward trick to not check the bounds dp[i+1]=dp[i]...
* use 'stop here' trick to handle negatives a[i]*b[j] + 0 instead of calling dfs(i+1,j+1)
* as we only accessing +-1, we can store just one recent row of dp instead of a full table

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(m)$$

#### Code

```kotlin
// 17ms
    fun maxDotProduct(a: IntArray, b: IntArray): Int {
        val d = Array(2) { IntArray(b.size+1) { -1000000 }}
        for (i in a.indices) for (j in b.indices) d[(i+1)%2][j+1] =
            maxOf(d[(i+1)%2][j], d[i%2][j+1], a[i]*b[j], a[i]*b[j]+d[i%2][j])
        return d[a.size%2][b.size]
    }
```
```rust
// 0ms
    pub fn max_dot_product(a: Vec<i32>, b: Vec<i32>) -> i32 {
        let mut d = vec![vec![-1000000;b.len()+1];2];
        for i in 0..a.len() { for j in 0..b.len() {
            d[(i+1)&1][j+1] = d[(i+1)&1][j].max(d[i&1][j+1]).max(a[i]*b[j]).max(a[i]*b[j]+d[i&1][j])
        }} d[a.len()&1][b.len()]
    }
```

