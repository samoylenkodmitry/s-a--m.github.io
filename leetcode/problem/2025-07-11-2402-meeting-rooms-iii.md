---
layout: leetcode-entry
title: "2402. Meeting Rooms III"
permalink: "/leetcode/problem/2025-07-11-2402-meeting-rooms-iii/"
leetcode_ui: true
entry_slug: "2025-07-11-2402-meeting-rooms-iii"
---

[2402. Meeting Rooms III](https://leetcode.com/problems/meeting-rooms-iii/description) hard
[blog post](https://leetcode.com/problems/meeting-rooms-iii/solutions/6945785/kotlin-rust-by-samoylenkodmitry-o3i3/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11072025-2402-meeting-rooms-iii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/f1zl4gPFCzI)
![1.webp](/assets/leetcode_daily_images/23dab087.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1046

#### Problem TLDR

Most frequent meeting room #hard #heap

#### Intuition

The corner case is `int overflow`.
Write the simulation:
* peek free room, or shift time to the first ending meeting
* purge all meetings until the time

#### Approach

* to remove the `time` variable, sort by the `room` number
* to use just a single queue, do the `rotation`: poll, shift time, push back; first free room is a queue size (interesting fact)
* to do without a queue: track ending times for each [100] room, peek the lowest in a linear time

#### Complexity

- Time complexity:
$$O(nlog(n))$$ or mnlog(n)

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 235ms
    fun mostBooked(n: Int, ms: Array<IntArray>): Int {
        ms.sortBy { it[0] }
        val freeRooms = PriorityQueue<Int>()
        val busyRooms = PriorityQueue<Pair<Long, Int>>(compareBy({it.first},{it.second}))
        for (r in 0..<n) freeRooms += r
        val freq = IntArray(n)
        for ((s, e) in ms) {
            while (busyRooms.size > 0 && busyRooms.peek().first <= s)
                freeRooms += busyRooms.poll().second
            if (freeRooms.size > 0) {
                val room = freeRooms.poll()
                ++freq[room]
                busyRooms += 1L * e to room
            } else {
                val (t, room) = busyRooms.poll()
                ++freq[room]
                busyRooms += (t + e - s) to room
            }
        }
        return freq.indexOf(freq.max())
    }

```
```kotlin

// 182ms
    fun mostBooked(n: Int, ms: Array<IntArray>): Int {
        ms.sortBy { it[0] }
        val q = PriorityQueue<Pair<Long, Int>>(compareBy({it.first},{it.second}))
        val freq = IntArray(n)
        for ((s, e) in ms)
            if (q.size > 0 && q.peek().first <= s || q.size >= n) {
                while (q.peek().first < s) q += 1L * s to q.poll().second
                val (end, room) = q.poll()
                ++freq[room]
                q += (1L * e + max(0, end - s)) to room
            } else {
                ++freq[q.size]
                q += 1L * e to q.size
            }
        return freq.indexOf(freq.max())
    }

```
```kotlin

// 181ms
    fun mostBooked(n: Int, ms: Array<IntArray>): Int {
        ms.sortBy { it[0] }; val freq = IntArray(n); val t = LongArray(n)
        for ((s, e) in ms) {
            val room = (0..<n).firstOrNull { t[it] <= s } ?: t.indexOf(t.min())
            t[room] = 1L * e + max(0, t[room] - s)
            ++freq[room]
        }
        return freq.indexOf(freq.max())
    }

```
```kotlin

// 173ms
    fun mostBooked(n: Int, ms: Array<IntArray>): Int {
        ms.sortBy { it[0] }
        val freeRooms = PriorityQueue<Int>()
        val busyRooms = PriorityQueue<Pair<Long, Int>>(compareBy{it.first})
        for (r in 0..<n) freeRooms += r
        val freq = IntArray(n); var t = 0L
        for ((s, e) in ms) {
            t = max(t, 1L * s)
            while (busyRooms.size > 0 && busyRooms.peek().first <= t)
                freeRooms += busyRooms.poll().second
            if (freeRooms.size < 1) {
                t = busyRooms.peek().first
                while (busyRooms.size > 0 && busyRooms.peek().first <= t)
                    freeRooms += busyRooms.poll().second
            }
            val room = freeRooms.poll()
            ++freq[room]
            busyRooms += (1L*e + (t - s)) to room
        }
        return freq.indexOf(freq.max())
    }

```

```kotlin

// 132ms
    fun mostBooked(n: Int, ms: Array<IntArray>): Int {
        ms.sortBy { it[0] }; val freq = IntArray(n); val t = LongArray(n)
        for ((s, e) in ms) {
            var min = Long.MAX_VALUE; var rmin = -1; var rs = -1
            for (r in 0..<n) {
                if (t[r] <= s) { rs = r; break }
                if (t[r] < min) { min = t[r]; rmin = r }
            }
            val room = max(rs, rmin)
            t[room] = 1L * e + max(0, t[room] - s)
            ++freq[room]
        }
        return freq.indexOf(freq.max())
    }

```
```rust

// 24ms
    pub fn most_booked(n: i32, mut ms: Vec<Vec<i32>>) -> i32 {
        ms.sort_unstable(); let n = n as usize;
        let (mut f, mut t, mut res) = (vec![0; n], vec![0; n], 0);
        for m in ms { let (s, e) = (m[0] as i64, m[1] as i64);
            let (mut rmin, mut rs, mut m) = (-1, -1, i64::MAX);
            for r in 0..n { if t[r] <= s { rs = r as i32; break }; if t[r] < m { rmin = r as i32; m = t[r] }}
            let room = rmin.max(rs) as usize; t[room] = e + 0.max(t[room] - s); f[room] += 1;
            if f[room] > f[res] { res = room } else if f[room] == f[res] { res = res.min(room) }
        }
        res as _
    }

```
```c++

// 63ms
int mostBooked(int n, std::vector<std::vector<int>>& ms) {
    sort(begin(ms), end(ms));
    long long t[100] = {}; int f[100] = {}, res = 0;
    for (auto& m : ms) {
        long long s = m[0], e = m[1], rs = -1, rmin = -1, mint = LONG_MAX;
        for (int r = 0; r < n; ++r)
            if (t[r] <= s) { rs = r; break; }
            else if (t[r] < mint) rmin = r, mint = t[r];
        int room = max(rs, rmin);
        t[room] = 1LL * e + max(0LL, t[room] - s);
        if (++f[room] > f[res] || (f[room] == f[res] && room < res)) res = room;
    }
    return res;
}

```

