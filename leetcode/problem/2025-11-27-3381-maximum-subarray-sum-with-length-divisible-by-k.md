---
layout: leetcode-entry
title: "3381. Maximum Subarray Sum With Length Divisible by K"
permalink: "/leetcode/problem/2025-11-27-3381-maximum-subarray-sum-with-length-divisible-by-k/"
leetcode_ui: true
entry_slug: "2025-11-27-3381-maximum-subarray-sum-with-length-divisible-by-k"
---

[3381. Maximum Subarray Sum With Length Divisible by K](https://leetcode.com/problems/maximum-subarray-sum-with-length-divisible-by-k/description/) medium
[blog post](https://leetcode.com/problems/maximum-subarray-sum-with-length-divisible-by-k/solutions/7377335/kotlin-rust-by-samoylenkodmitry-7tye/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27112025-3381-maximum-subarray-sum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/xNh7cuCHvxg)

![83052778-c9b7-4dde-adc0-eed850cd8a2b (1).webp](/assets/leetcode_daily_images/d1e3108f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1186

#### Problem TLDR

Max sum of %k-length subarray #medium

#### Intuition

```j
    // -1 1 -1 1 -1 1   %3
    //    *
    //    *  * *
    //
    // all sums O(n^2)
    // all subarrays O(n^2) k=1
    // find just max subarray sum, then expand/shrink?
    // use two pointers and shrink from both ends until %k?
    // 2 -5 3 2 1     %3, which pointer to move?
    // i        j
    //
    //      i    ending at i we have i/k possible starting points
    //           all of them are moving with i + 1,
    //  **s**s**s*i  %3
    //          ***
    //       ******
    //    *********
    //                      the algo is O(n*(n/k)) = O(n^2)
    //
    // looks like a hard problem

    // 17 minutes, use hint - min prefix sum ending at every i%k (didn't tell much)
```

* dp[i] = sum(i-k..i)+max(0, dp[i-k]), where dp[i] is answer for array ending at i

#### Approach

* solution from lee (i copied it in rust) is uncomprehensible, let's just say it is compressed

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 44ms
    fun maxSubarraySum(n: IntArray, k: Int): Long {
        val dp = LongArray(n.size); val p = LongArray(n.size+1)
        for (i in n.indices) p[i+1] = n[i].toLong()+p[i]
        for (i in n.indices) dp[i] = (p[i+1]-if(i-k+1>=0)p[i-k+1]else 0) +
                                      max(0, if(i-k+1>=k)dp[i-k]else 0)
        return dp.drop(k-1).max()
    }
```
```rust
// 7ms
    pub fn max_subarray_sum(n: Vec<i32>, k: i32) -> i64 {
        let k = k as usize; let (mut p, mut s) = (vec![i64::MAX/2; k], 0); p[k-1] = 0;
        (0..n.len()).map(|i| { s += n[i] as i64;
            let r = s - p[i%k]; p[i%k] = s.min(p[i%k]); r}).max().unwrap()
    }
```

