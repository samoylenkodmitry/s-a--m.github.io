---
layout: leetcode-entry
title: "2141. Maximum Running Time of N Computers"
permalink: "/leetcode/problem/2023-07-27-2141-maximum-running-time-of-n-computers/"
leetcode_ui: true
entry_slug: "2023-07-27-2141-maximum-running-time-of-n-computers"
---

[2141. Maximum Running Time of N Computers](https://leetcode.com/problems/maximum-running-time-of-n-computers/description/) hard
[blog post](https://leetcode.com/problems/maximum-running-time-of-n-computers/solutions/3822065/kotlin-how-to-use-time/)
[substack](https://dmitriisamoilenko.substack.com/p/27072023-2141-maximum-running-time?sd=pf)
![image.png](/assets/leetcode_daily_images/0546d6e6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/288

#### Problem TLDR

Maximum time to use `n` batteries in parallel

#### Hint 1

Batteries `5 5 5` is equal to `1 2 3 4 5` to run `3` computers for `5` minutes.

#### Hint 2

Batteries are swapped instantly, so we can drain all `1 2 3 4 5` with just `3` computers, but if a pack is `1 2 3 4 100` we can only drain `5` from the last `100` battery. (or less)

#### Hint 3

Energy of `5 5 5` is `15` to run for `5` minutes.
Energy in `1 2 3 4 100` is `1+2+3+4+5` when run for `5` minutes.
Energy in `1 2 3 4 100` is `1+2+3+4+4` when run for `4` minutes.
Energy in `1 2 3 4 100` is `1+2+3+3+3` when run for `3` minutes.

#### Intuition

The Binary Search idea is first to mind, as with growth of run time the function of `canRun` do the flip.

However, to detect if we `canRun` the given `time` is not so trivial.

We can use all batteries by swapping them every minute. To use `5` batteries in `3` computers, we can first use the max capacity and change others:

```

1 2 3 4 5
    1 1 1
    1 1 1
    1 1 1
  1   1 1
1 1     1

```

In this example, `time = 5`. Or we can have just `3` batteries with capacity of `5` each: `5 5 5`. What if we add another battery:

```

1 2 3 4 5 9
      1 1 1
      1 1 1
      1 1 1
      1 1 1
    1   1 1
  1 1     1
  1 1     1

```

`Time` becomes `7`, or we can have `7 7 7` battery pack with total `energy = 3 * 7 = 21`. And we don't use `1` yet.

Let's observe the energy for the `time = 7`:

```

1 2 3 4 5 9
* 1 1 1 1 1
  1 1 1 1 1
    1 1 1 1
      1 1 1
        1 1
          1
          1
```

We didn't use `1`, but had we another `1` the total energy will be `21 + 1 + 1 + 1(from 9)` or `24`, which is equal to `3 * 8`, or `time = 8`.
So, by this diagram, we can take at most `time` power units from each battery.
So, our function `canRun(time)` is: `energy(time) >= time * n`. Energy is a sum of all batteries running at most `time`.

#### Approach

Binary Search:
* inclusive `lo` & `hi`
* last check `lo == hi`
* compute result `res = mid`
* boundaries `lo = mid + 1`, `hi = mid - 1`

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```

    fun maxRunTime(n: Int, batteries: IntArray): Long {
        // n=3       1 2 3 4 5 6 7 9
        // time = 4
        // we need 4 4 4, take 1 2 3 4 4 4 4 4
        // time = 5
        // we need 5 5 5, take 1 2 3 4 5 5 5 5

        // n=3         3 3 3 80
        // time = 1    1 1 1 1      vs    1 1 1
        // time = 2    2 2 2 2      vs    2 2 2
        // time = 3    3 3 3 3      vs    3 3 3
        // time = 4    3 3 3 4 (13) vs    4 4 4 (16)
        // time = 5    3 3 3 5 (14) vs    5 5 5 (15)
        // time = 6    3 3 3 6 (15) vs    6 6 6 (18)
        var lo = 0L
        var hi = batteries.asSequence().map { it.toLong() }.sum() ?: 0L
        var res = 0L
        while (lo <= hi) {
          val mid = lo + (hi - lo) / 2L
          val canRun = n * mid <= batteries.asSequence().map { minOf(it.toLong(), mid) }.sum()!!
          if (canRun) {
            res = mid
            lo = mid + 1L
          } else hi = mid - 1L
        }
        return res
    }

```

