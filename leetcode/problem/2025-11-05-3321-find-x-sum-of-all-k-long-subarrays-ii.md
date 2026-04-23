---
layout: leetcode-entry
title: "3321. Find X-Sum of All K-Long Subarrays II"
permalink: "/leetcode/problem/2025-11-05-3321-find-x-sum-of-all-k-long-subarrays-ii/"
leetcode_ui: true
entry_slug: "2025-11-05-3321-find-x-sum-of-all-k-long-subarrays-ii"
---

[3321. Find X-Sum of All K-Long Subarrays II](https://leetcode.com/problems/find-x-sum-of-all-k-long-subarrays-ii/description/) hard
[blog post](https://leetcode.com/problems/find-x-sum-of-all-k-long-subarrays-ii/solutions/7328235/kotlin-by-samoylenkodmitry-tcyy/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05112025-3321-find-x-sum-of-all-k?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/i8wup3zXawU)

![169218bf-5867-48db-b3f0-d59fbfeaebe6 (1).webp](/assets/leetcode_daily_images/45d21649.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1164

#### Problem TLDR

Sums of x most frequent from k-windows #hard

#### Intuition

The main hardness is how to maintain the `top sorted X values` in a sliding window.
There is a trick from:
* use TreeSet for the X values
* when removing from X, place removed numbers in a second TreeSet B
* balance if the best number from B is better than lowest from X

```j
    // 3 3 1 3 3
    // how to speed up/re-use the sum of most frequent
    // 1 1 2 2 3 4 2 3
    // 1 1 2 2 3 4        2 1 4 3   top 2 is 2 1 or sum(1 1 2 2)
    //   1 2 2 3 4 2      2 4 3 1   top 2 is 2 4 or sum(2 2 2 4)   -1 +2 binarysearch n is in topX?
    //     2 2 3 4 2 3    2 3 4 2   top 2 is 2 3 or sum(2 2 2 3 3) -4(all) + 3(all)
    // how to find which values are out of top
    // the add to top is simple: only when frequency increases, same value can became in top
    // but what values are out of top? maybe the last in the top

    // can we only keep X values in set? - no, because we loose some promising big numbers like 4 from 1,1,2,2,3,4,2,3
    // 11122333  x=1
    // 111
    //  112
    //   122
    // time 55, look for hint: the misteriuos second set
```

#### Approach

* related problem: https://leetcode.com/problems/sliding-window-median/description/

#### Complexity

- Time complexity:
$$O(nlogx)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 1054ms
    fun findXSum(n: IntArray, k: Int, x: Int): LongArray {
        val f = HashMap<Int, Long>(); val res = LongArray(n.size-k+1)
        val q = TreeSet<Int>(compareBy<Int>{f[it]?:0L}.thenBy<Int>{it})
        val o = TreeSet<Int>(q.comparator()); var sum = 0L
        fun poll() = if (q.size > x) {
                sum -= 1L * (f[q.first()] ?: 0) * q.first()
                o += q.pollFirst()
            } else Unit
        fun add(n: Int) { q += n; sum += 1L * f[n]!! * n; poll() }
        fun balance() {
            poll()
            if (o.size > 0 && (q.size < x || q.comparator().compare(o.last(),q.first()) > 0)) add(o.pollLast())
        }
        for (i in n.indices) {
            val oldF = f[n[i]] ?: 0L; val newF = 1L + oldF
            if (n[i] in q) sum -= oldF * n[i]
            q -= n[i]; o -= n[i]; f[n[i]] = newF; add(n[i])
            if (i >= k-1) {
                balance()
                res[i-k+1] = sum
                val n = n[i-k+1]; val oldF = f[n] ?: 0L; val newF = oldF - 1L
                if (n in q) sum -= oldF * n
                o -= n; q -= n; f[n] = newF; if (newF > 0L) add(n) else f -= n
            }
        }
        return res
    }

```

