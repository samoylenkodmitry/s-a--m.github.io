---
layout: leetcode-entry
title: "2226. Maximum Candies Allocated to K Children"
permalink: "/leetcode/problem/2025-03-14-2226-maximum-candies-allocated-to-k-children/"
leetcode_ui: true
entry_slug: "2025-03-14-2226-maximum-candies-allocated-to-k-children"
---

[2226. Maximum Candies Allocated to K Children](https://leetcode.com/problems/maximum-candies-allocated-to-k-children/description) medium
[blog post](https://leetcode.com/problems/maximum-candies-allocated-to-k-children/solutions/6534985/kotlin-rust-by-samoylenkodmitry-qtoe/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14032025-2226-maximum-candies-allocated?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/DbrKzeIlU0U)
![1.webp](/assets/leetcode_daily_images/1b1d9899.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/927

#### Problem TLDR

Distribute candies to k equal buckets #medium #binary_search

#### Intuition

Binary search in a space of k: check range 1..max(candies), try to take candies to the sum. If it possible, try a bigger pile.

Follow up: what if candies is already sorted? Is there O(n) algorithm? (I can't find it)

#### Approach

* calculate result in a separate variable res = max(res, m) to be safe, but you can golf and just use the `hi` variable as answer

#### Complexity

- Time complexity:
$$O(nlog(max))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maximumCandies(candies: IntArray, k: Long): Int {
        var lo = 1; var hi = candies.max()
        while (lo <= hi) {
            val m = lo + (hi - lo) / 2
            if (k <= candies.sumOf { 1L * it / m })
            lo = m + 1 else hi = m - 1
        }
        return hi
    }

```
```rust

    pub fn maximum_candies(candies: Vec<i32>, k: i64) -> i32 {
        let (mut lo, mut hi) = (1, 10_000_000);
        while lo <= hi {
            let m = lo + (hi - lo) / 2;
            if k <= candies.iter().map(|&x| x as i64 / m).sum()
            { lo = m + 1 } else { hi = m - 1 }
        } hi as _
    }

```
```c++

    int maximumCandies(vector<int>& candies, long long k) {
        int lo = 1, hi = 10000000;
        while (lo <= hi) {
            int m = lo + (hi - lo) / 2; long s = 0;
            for (int c: candies) s += 1L * c / m;
            k > s ? hi = m - 1 : lo = m + 1;
        } return hi;
    }

```

