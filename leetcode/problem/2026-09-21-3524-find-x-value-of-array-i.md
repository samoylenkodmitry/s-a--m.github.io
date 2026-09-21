---
layout: leetcode-entry
title: "3524. Find X Value of Array I"
permalink: "/leetcode/problem/2026-09-21-3524-find-x-value-of-array-i/"
leetcode_ui: true
entry_slug: "2026-09-21-3524-find-x-value-of-array-i"
---

[3524. Find X Value of Array I](https://leetcode.com/problems/find-x-value-of-array-i/solutions/8532727/kotlin-rust-by-samoylenkodmitry-3wia/) medium
[substack](https://dmitriisamoilenko.substack.com/p/21092026-3524-find-x-value-of-array?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/d1IseeJYev8)

https://dmitrysamoylenko.com/leetcode/

![21.09.2026.webp](/assets/leetcode_daily_images/21.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1489

#### Problem TLDR

Count subarrays product%k = x for x in 0..<k

#### Intuition

Track count of subarrays so far in [k] array. The new reminder after adding n to each subarray for each x in 0..<k is the v=x*n%k. Increment all the subarrays ending by v cnt[v] += cnt[i] by count of all subarrays ending by i that now added number n to them making the product reminder i*n%k

#### Approach

* look at others people answers i guess

#### Complexity

- Time complexity:
$$O(nk)$$

- Space complexity:
$$O(k)$$

#### Code

```kotlin
    fun resultArray(n: IntArray, k: Int) = LongArray(k).also { res ->
        var c = IntArray(k)
        for (x in n) c = IntArray(k).also {
            it[x % k]++; res[x % k]++
            for (i in 0..<k) {
                val v = (1L * i * x % k).toInt()
                it[v] += c[i]; res[v] += c[i]
            }
        }
    }
```
```rust
    pub fn result_array(n: Vec<i32>, k: i32) -> Vec<i64> {
        let k = k as usize; let (mut res, mut c) = (vec![0; k], vec![0; k]);
        for x in n {
            let (mut next, x) = (vec![0; k], x as usize % k);
            next[x] += 1; res[x] += 1;
            for (i, cnt) in (0..).zip(c) {
                let v = i * x % k; next[v] += cnt; res[v] += cnt as i64
            } c = next
        } res
    }
```

