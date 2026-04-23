---
layout: leetcode-entry
title: "2402. Meeting Rooms III"
permalink: "/leetcode/problem/2024-02-18-2402-meeting-rooms-iii/"
leetcode_ui: true
entry_slug: "2024-02-18-2402-meeting-rooms-iii"
---

[2402. Meeting Rooms III](https://leetcode.com/problems/meeting-rooms-iii/description) hard
[blog post](https://leetcode.com/problems/meeting-rooms-iii/solutions/4745785/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18022024-2402-meeting-rooms-iii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/q3nIjTzhYHw)
![image.png](/assets/leetcode_daily_images/66ee08c5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/510

#### Problem TLDR

Most frequent room of 0..<n where each meeting[i]=[start, end) takes or delays until first available.

#### Intuition

Let's observe the process of choosing the room for each meeting:

```bash
    // 0 1     0,10 1,5 2,7 3,4
    //10       0,10
    //   5          1,5
    //                  2,7
    //   10             5,10=5+(7-2)
    //                       3,4
    //11                    10,11

    // 0 1 2    1,20  2,10  3,5  4,9  6,8
    //20        1,20
    //  10            2,10
    //     5                3,5
    //                           4,9
    //    10                     5,10
    //                                6,8
    //  12                           10,12

    //  0  1  2  3  18,19  3,12  17,19  2,13  7,10
    //               2,13  3,12   7,10 17,19 18,19
    // 13            2,13
    //    12               3,12
    //       10                   7,10
    //          19                     17,19
    //     <-19                               18,19
    //  1  1  2  1

    // 0  1  2  3   19,20 14,15 13,14 11,20
    //              11,20 13,14 14,15 19,20
    //20              *
    //   14                 *
    //    <-15
```

Some caveats are:
* we must take room with lowest index
* this room must be empty or meeting must already end
* the interesting case is when some rooms are still empty, but some already finished the meeting.

To handle finished meetings, we can just repopulate the PriorityQueue with the current time.

#### Approach

Let's try to write a minimal code implementation.
* Kotiln heap is a min-heap, Rust is a max-heap
* Kotlin `maxBy` is not greedy, returns first max. Rust `max_by_key` is greedy and returns the last visited max, so not useful here.

#### Complexity

- Time complexity:
$$O(mnlon(n))$$, `m` is a meetings size. Repopulation process is `nlog(n)`. Just finding the minimum is O(mn).

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun mostBooked(n: Int, meetings: Array<IntArray>): Int {
    meetings.sortWith(compareBy { it[0] })
    val v = LongArray(n); val freq = IntArray(n)
    for ((s, f) in meetings) {
      val room = (0..<n).firstOrNull { v[it] <= s } ?: v.indexOf(v.min())
      if (v[room] > s) v[room] += (f - s).toLong() else v[room] = f.toLong()
      freq[room]++
    }
    return freq.indexOf(freq.max())
  }

```
```rust

    pub fn most_booked(n: i32, mut meetings: Vec<Vec<i32>>) -> i32 {
      let (mut v, mut freq) = (vec![0; n as usize], vec![0; n as usize]);
      meetings.sort_unstable();
      for m in meetings {
        let (s, f) = (m[0] as i64, m[1] as i64);
        let room = v.iter().position(|&v| v <= s).unwrap_or_else(|| {
          let min = *v.iter().min().unwrap();
          v.iter().position(|&v| v == min).unwrap() });
        freq[room] += 1;
        v[room] = if v[room] > s { f - s + v[room] } else { f }
      }
      let max = *freq.iter().max().unwrap();
      freq.iter().position(|&f| f == max).unwrap() as i32
    }

```

