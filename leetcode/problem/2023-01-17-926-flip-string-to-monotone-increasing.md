---
layout: leetcode-entry
title: "926. Flip String to Monotone Increasing"
permalink: "/leetcode/problem/2023-01-17-926-flip-string-to-monotone-increasing/"
leetcode_ui: true
entry_slug: "2023-01-17-926-flip-string-to-monotone-increasing"
---

[926. Flip String to Monotone Increasing](https://leetcode.com/problems/flip-string-to-monotone-increasing/description/) medium

[https://t.me/leetcode_daily_unstoppable/88](https://t.me/leetcode_daily_unstoppable/88)

[blog post](https://leetcode.com/problems/flip-string-to-monotone-increasing/solutions/3062530/kotlin-dp/)

```kotlin
    fun minFlipsMonoIncr(s: String): Int {
        // 010110  dp0  dp1    min
        // 0       0    0      0
        //  1      1    0      1
        //   0     1    1      1
        //    1    2    1      1
        //     1   3    1      1
        //      0  3    2      2
        var dp0 = 0
        var dp1 = 0

        for (i in 0..s.lastIndex) {
            dp0 = if (s[i] == '0') dp0 else 1 + dp0
            dp1 = if (s[i] == '1') dp1 else 1 + dp1
            if (dp0 <= dp1) dp1 = dp0
        }

        return minOf(dp0, dp1)
    }

```

We can propose the following rule: let's define `dp0[i]` is a min count of flips from `1` to `0` in the `0..i` interval.
Let's also define `dp1[i]` is a min count of flips from `0` to `1` in the `0..i` interval.
We observe that `dp0[i] = dp0[i-1] + (flip one to zero? 1 : 0)` and `dp1[i] = dp1[i-1] + (flip zero to one? 1 : 0)`.
One special case: if on the interval `0..i` one-to-zero flips count is less than zero-to-one then we prefer to flip everything to zeros, and `dp1[i]` in that case becomes `dp0[i]`.

Just write down what is described above.
* dp arrays can be simplified to single variables.

Space: O(1), Time: O(N)

