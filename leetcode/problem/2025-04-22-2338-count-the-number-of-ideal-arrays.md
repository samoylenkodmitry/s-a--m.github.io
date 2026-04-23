---
layout: leetcode-entry
title: "2338. Count the Number of Ideal Arrays"
permalink: "/leetcode/problem/2025-04-22-2338-count-the-number-of-ideal-arrays/"
leetcode_ui: true
entry_slug: "2025-04-22-2338-count-the-number-of-ideal-arrays"
---

[2338. Count the Number of Ideal Arrays](https://leetcode.com/problems/count-the-number-of-ideal-arrays/description/) hard
[blog post](https://leetcode.com/problems/count-the-number-of-ideal-arrays/solutions/6676406/kotlin-by-samoylenkodmitry-nd7e/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22042025-2338-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/zAXhbfWrc6c)
![1.webp](/assets/leetcode_daily_images/c41078bb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/966

#### Problem TLDR

Arrays a[i] % a[i - 1] == 0, i..n, a[i]..max #hard #combinatorics

#### Intuition

Didn't solve. And didn't understand the solution.
To make it work you have to be fluent with combinatorics.
You have to be fluent with `Stars and bars` https://cp-algorithms.com/combinatorics/stars_and_bars.html.

My thoughts rundown is irrelevant here, so I will not post it.

Some thoughts about the solution:
* arrays are `aaa | bbb | ccc`, where `|` is the bars. `1 | 2 | 4 4` or `1 1| 2 |4` or `1 | 2 2 | 4`.
* the max *uniq* sequence length is for `2`: `1,2,4,8,2^4,2^5,...2^i,..10000`, max i is 2^13=8192 < 10000
* res += `n choose k`, n in `1..maxValue`, k in `0..13`. We considering placing `1..maxValue` numbers into a length of `0..13` places

#### Approach

* maybe I should try more combinatorics problems to better understand them; right now they are not picturing in my brain canvas

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(nlog(n))$$

#### Code

```kotlin

    fun idealArrays(n: Int, maxValue: Int): Int {
        val comb = Array(10001) { IntArray(14) }; val cnt = Array(10001) { IntArray(14) }
        val M = 1_000_000_007; comb[0][0] = 1; var res = 0L
        for (s in 1..10000) { comb[s][0] = 1
            for (r in 1..13) comb[s][r] = (comb[s - 1][r - 1] + comb[s - 1][r]) % M }
        for (div in 1..10000) {
            ++cnt[div][0]
            for (i in 2 * div..10000 step div)
                for (bars in 0..12) cnt[i][bars + 1] += cnt[div][bars]
        }
        for (i in 1..maxValue) for (bars in 0..min(13, n))
            res = (1L * cnt[i][bars] * comb[n - 1][bars] + res) % M
        return res.toInt()
    }

```

