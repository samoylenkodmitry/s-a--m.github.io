---
layout: leetcode-entry
title: "2364. Count Number of Bad Pairs"
permalink: "/leetcode/problem/2025-02-09-2364-count-number-of-bad-pairs/"
leetcode_ui: true
entry_slug: "2025-02-09-2364-count-number-of-bad-pairs"
---

[2364. Count Number of Bad Pairs](https://leetcode.com/problems/count-number-of-bad-pairs/description/) medium
[blog post](https://leetcode.com/problems/count-number-of-bad-pairs/solutions/6397416/kotlin-rust-by-samoylenkodmitry-4j0k/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09022025-2364-count-number-of-bad?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/yw-CX03h0IQ)
![1.webp](/assets/leetcode_daily_images/73beb107.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/890

#### Problem TLDR

Count pairs a[i] - a[j] != j - i #medium #counting #sorting

#### Intuition

Staring blindly into a void of emptiness:

```j

    // 0 1 2 3
    // 4 1 3 3
    //     *   (expect: 1-2, 0-1)
    //       * (expect: 1-1*, 2-2)
    //
    // 1: 1     ->3(3)
    // 3: 2, 3  ->4(3)
    // 4: 0     ->5(1), 6(2), 7(3)
    //
    // 4-1, 4-3, 4-3
    //   1-3, 1-3(good)
    //      3-3
    //
    //
    // * 5 6 7
    //   * 2 3
    //     * 4
    // x, x+1, x+2, ...

```

Hoping to uncover the truth:

```j

    // j - i = nums[j] - nums[i]
    // j - nums[j] = i - nums[i]

```

I couldn't solve it without the hint. Every approach led to dead ends and cold, lifeless patterns of O(n^2). Failed and humbled, I resorted to the hint.

Now, it was only a matter of stacking the right tricks - like puzzle pieces clicking into place. A hashmap counter, a running sum of frequencies. Simple tools, deadly in the right hands.

#### Approach

* They weren't kidding about the Long's.
* `1L *` is shorter than `.toLong()` sometimes.
* The total is `n * (n - 1) / 2` or we can count the running total `+= i`.
* Ever had that feeling when you think you know something, but when you look at it again, it's something entirely different? That's the solution without a HashMap: sort differences and scan them linearly to count frequencies.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun countBadPairs(nums: IntArray): Long {
        val f = HashMap<Int, Long>()
        return nums.withIndex().sumOf { (i, n) ->
            val s = f[i - n] ?: 0; f[i - n] = 1 + s; i - s
        }
    }

```
```rust

    pub fn count_bad_pairs(mut n: Vec<i32>) -> i64 {
        for i in 0..n.len() { n[i] -= i as i32 }; n.sort_unstable();
        n.len() as i64 * (n.len() as i64 - 1) / 2 -
        n.chunk_by(|a, b| a == b).map(|c| (c.len() * (c.len() - 1) / 2) as i64).sum::<i64>()
    }

```
```c++

    long long countBadPairs(vector<int>& n) {
        long long r = 0, f = 0, m = size(n);
        for (int i = 0; i < m; ++i) n[i] -= i;
        sort(begin(n), end(n));
        for (int i = 0; i < m; ++i)
            r += i - f, ++f *= i + 1 < m && n[i] == n[i + 1];
        return r;
    }

```

