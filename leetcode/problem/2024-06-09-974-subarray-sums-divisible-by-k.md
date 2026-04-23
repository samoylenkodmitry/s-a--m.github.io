---
layout: leetcode-entry
title: "974. Subarray Sums Divisible by K"
permalink: "/leetcode/problem/2024-06-09-974-subarray-sums-divisible-by-k/"
leetcode_ui: true
entry_slug: "2024-06-09-974-subarray-sums-divisible-by-k"
---

[974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/description/) medium
[blog post](https://leetcode.com/problems/subarray-sums-divisible-by-k/solutions/5281959/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09062024-974-subarray-sums-divisible?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/STAWZH_FmLc)
![2024-06-09_06-36_1.webp](/assets/leetcode_daily_images/3e01838e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/634

#### Problem TLDR

Count subarrays divisible by `k` #medium #hashmap

#### Intuition

Let's observe an example:

```j

    //   0 1 2  3  4 5
    //   4 5 0 -2 -3 1   s  k=5      sums    count
    //i                              0 -> 1
    //   i               4  4%5=4    4 -> 1
    //     i             9  9%5=4    4 -> 2  +1
    //       i           9  9%5=4    4 -> 3  +2
    //          i        7  7%5=2    2 -> 1
    //             i     4  4%5=4    4 -> 4  +3
    //               i   5  5%5=0    0 -> 2  +1
```
We can compute the `running sum`. Subarray sum can be computed from the previous running sum: `sum[i..j] = sum[0..j] - sum[0..i]`. Next, if sum is divisibile by `k`, then we can apply `%` operation rule: `sum[i..j] % k = 0 = sum[0..j] % k - sum[0..i] % k`, or in another words: `sum[0..i] % k == sum[0..j] % k`. So, we just need to keep track of all the remiders.

Corner case is when subarray is starts with first item, just make a sentinel counter for it: `sums[0] = 1`.

#### Approach

* using iterators saves some lines of code
* did you know about `hashMapOf` & `HashMap::from` ?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(k)$$

#### Code

```kotlin

    fun subarraysDivByK(nums: IntArray, k: Int): Int {
        val sums = hashMapOf(0 to 1); var s = 0
        return (0..<nums.size).sumOf { i ->
            s = (s + nums[i] % k + k) % k
            val count = sums[s] ?: 0
            sums[s] = 1 + count
            count
        }
    }

```
```rust

    pub fn subarrays_div_by_k(nums: Vec<i32>, k: i32) -> i32 {
        let (mut sums, mut s) = (HashMap::from([(0, 1)]), 0);
        (0..nums.len()).map(|i| {
            s = (s + nums[i] % k + k) % k;
            let count = *sums.entry(s).or_default();
            sums.insert(s, 1 + count);
            count
        }).sum()
    }

```

