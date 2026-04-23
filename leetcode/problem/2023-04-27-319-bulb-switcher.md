---
layout: leetcode-entry
title: "319. Bulb Switcher"
permalink: "/leetcode/problem/2023-04-27-319-bulb-switcher/"
leetcode_ui: true
entry_slug: "2023-04-27-319-bulb-switcher"
---

[319. Bulb Switcher](https://leetcode.com/problems/bulb-switcher/description/) medium

```kotlin

fun bulbSwitch(n: Int): Int {
    if (n <= 1) return n
    var count = 1
    var interval = 3
    var x = 1
    while (x + interval <= n) {
        x = x + interval
        interval += 2
        count++
    }
    return count
}

```

[blog post](https://leetcode.com/problems/bulb-switcher/solutions/3459491/kotlin-spot-the-pattern/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/leetcode-daily-27042023)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/193
#### Intuition
Let's draw a diagram and see if any pattern here:

```

//      1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
//
// 1    1 1 1 1 1 1 1 1 1  1 1  1  1  1  1  1  1  1  1
// 2      0 . 0 . 0 . 0 .  0 .  0  .  0  .  0  .  0  .
// 3        0 . . 1 . . 0  . .  1  .  .  0  .  .  1  .
// 4          1 . . . 1 .  . .  0  .  .  .  1  .  .  .
// 5            0 . . . .  1 .  .  .  .  1  .  .  .  .
// 6              0 . . .  . .  1  .  .  .  .  .  0  .
// 7                0 . .  . .  .  .  1  .  .  .  .  .
// 8                  0 .  . .  .  .  .  .  0  .  .  .
// 9                    1  . .  .  .  .  .  .  .  1  .
// 10                      0 .  .  .  .  .  .  .  .  .
// 11                        0  .  .  .  .  .  .  .  .
// 12                           0  .  .  .  .  .  .  .
// 13                              0  .  .  .  .  .  .
// 14                                 0  .  .  .  .  .
// 15                                    0  .  .  .  .
// 16                                       1  .  .  .
// 17                                          0  .  .
// 18                                             0  .
// 19                                                0

```

One rule is: number of switches for each new value is a number of divisors.
Another rule: we can reuse the previous result.
However, those rules didn't help much, let's observe another pattern: `diagonal sequence have increasing intervals of zeros by 2`

#### Approach
Use found law and write the code.
#### Complexity
- Time complexity:
That is tricky, let's derive it:
$$
n = 1 + 2 + (1+2+2) + (1+2+2+2) + (...) + (1+2k)
$$, or
$$
n = \sum_{i=0}^{k}1+2i = k(1 + 2 + 1 + 2k)/2
$$, then count of elements in arithmetic progression `k` is:
$$
O(k) = O(\sqrt{n})
$$, which is our time complexity.
- Space complexity:
$$O(1)$$

