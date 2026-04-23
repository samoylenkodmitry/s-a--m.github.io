---
layout: leetcode-entry
title: "2466. Count Ways To Build Good Strings"
permalink: "/leetcode/problem/2024-12-30-2466-count-ways-to-build-good-strings/"
leetcode_ui: true
entry_slug: "2024-12-30-2466-count-ways-to-build-good-strings"
---

[2466. Count Ways To Build Good Strings](https://leetcode.com/problems/count-ways-to-build-good-strings/description/) medium
[blog post](https://leetcode.com/problems/count-ways-to-build-good-strings/solutions/6204776/kotlin-rust-by-samoylenkodmitry-13wt/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30122024-2466-count-ways-to-build?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6kOm_LN8iFM)
[deep-dive](https://notebooklm.google.com/notebook/be75821b-01b8-4a64-8530-23d182e5c604/audio)
![1.webp](/assets/leetcode_daily_images/63196ac2.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/848

#### Problem TLDR

Ways to make `01`-string length of `low..high` #medium #dynamic_programming

#### Intuition

Let's observe what happens when we adding zeros and ones:

```j

    // "0" "11" -> 00  011   110 1111
    //  00 111  -> 00 111 00111  111111 0000 11100
    //  000 111 -> 000 111 000111 111000 000000 111111
    //                     *      .
    //                     000111000 000111111
    //                            .
    //                            111000000 111000111

```

* each new string is a start of another tree of possibilities
* only the length of this string matters

Let's do a full Depth-First search, give the current length add zeros or add ones and count the total ways. The result can be cached by the key of the starting length.

#### Approach

* top-down can be rewritten to the bottom-up DP
* then we can reverse the order of iteration

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    val dp = HashMap<Int, Long>()
    fun countGoodStrings(low: Int, high: Int, zero: Int, one: Int, len: Int = 0): Long =
        if (len > high) 0L else dp.getOrPut(len) {
            val addZeros = countGoodStrings(low, high, zero, one, len + zero)
            val addOnes = countGoodStrings(low, high, zero, one, len + one)
            ((if (len < low) 0 else 1) + addZeros + addOnes) % 1_000_000_007L
        }

```
```rust

    pub fn count_good_strings(low: i32, high: i32, zero: i32, one: i32) -> i32 {
        let mut dp = vec![0; 1 + high as usize];
        for len in 0..=high as usize {
            let add_zeros = dp.get(len - zero as usize).unwrap_or(&0);
            let add_ones = dp.get(len - one as usize).unwrap_or(&0);
            let curr = (low + len as i32 <= high) as usize;
            dp[len] = (curr + add_zeros + add_ones) % 1_000_000_007
        }; dp[high as usize] as i32
    }

```
```c++

    int countGoodStrings(int low, int high, int zero, int one) {
        int dp[100001];
        for (int l = 0; l <= high; ++l) dp[l] = ((low + l <= high) +
            (l < zero ? 0 : dp[l - zero]) +
            (l < one ? 0 : dp[l - one])) % 1000000007;
        return dp[high];
    }

```

