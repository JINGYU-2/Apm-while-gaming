`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/08/31 12:03:24
// Design Name: 
// Module Name: top
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module top (
    input wire clk,            
    input wire restart,        
    output wire trig,          
    input wire echo,           
    output wire led,          
    output reg led1,
    output reg led2,
    output reg led3,
    output reg led4,
    output wire tx,            
    input wire rx
);

    wire [32:0] distance;  // Wire to hold the distance measured by the sonic module
    reg [7:0] ascii_data;  // ASCII data to be sent via UART
    reg [3:0] digit_index; // Track which ASCII character to send
    wire busy;             // UART busy signal
    wire [7:0] received_data;
    reg send_uart;         // Signal to start UART transmission
    reg [32:0] distance_copy;

    // Instantiate the ultrasonic sensor module
    sonic u_sonic (
        .clock(clk),
        .restart(restart),
        .trig(trig),
        .echo(echo),
        .led(led),
        .distance(distance)
    );

    // UART transmission control
    always @(posedge clk) begin
        if (!busy && !send_uart) begin
            distance_copy <= distance;  // Take a snapshot of distance
            digit_index <= 0;
            send_uart <= 1'b1; // Start sending the distance as ASCII
        end else if (send_uart && !busy) begin
            case (digit_index)
                0: ascii_data <= ((distance_copy / 100) % 10)+48;  // Hundreds digit
                1: ascii_data <= ((distance_copy / 10) % 10)+48;  // Tens digit
                2: ascii_data <= (distance_copy % 10)+48;    // Ones digit
                3: ascii_data <= 8'h0A;  // Newline (LF) 
                4: ascii_data <= 8'h0D;  
            endcase
            digit_index <= digit_index + 1;
            if (digit_index == 4)
                send_uart <= 1'b0; // Stop after sending the distance
        end
    end
   
    uart_tx_module u_uart_tx (
        .clk(clk),
        .reset(restart),
        .data(ascii_data),
        .send(send_uart),
        .tx(tx),
        .busy(busy)
    );
    
        uart_rx_module u_uart_rx (
        .clk(clk),
        .reset(restart),
        .rx(rx),
        .data(received_data),
        .ready(uart_ready)
    );
     
    reg [35:0] counter = 0;
    reg [35:0] counter1 = 0;
    reg [35:0] counter2 = 0;
    parameter first = 10; // 5 minutes in clock cycles
   /* reg [35:0] counter1 = 0;
    parameter TEN_MIN_CYCLES = 6000000000; // 10 minutes in clock cycles
    reg [35:0] counter2 = 0;
    parameter FIFTEEN_MIN_CYCLES = 9000000000; // 15 minutes in clock cycles
    reg [35:0] counter3 = 0;
    parameter TWENTY_MIN_CYCLES = 12000000000; // 20 minutes in clock cycles */

  always @(posedge clk) begin
    if (restart) begin
        counter <= 0;
        counter1 <= 0;
        counter2 <= 0;
        led1 <= 0;
        led2 <= 0;
        led3 <= 0;
        led4 <= 0;
    end 
    else begin
        counter <= counter + 1;
        if (counter == 27'b101111101011110000100000000) begin  // 100 million in binary
            counter1 <= counter1 + 1;
            counter <= 0;
        end

        if (counter1 == 6'b111100) begin  // 60 
            counter1 <= 0;
            
            if (counter2 < 6'b101000) begin  // 40
                led1 <= 1;
                led2 <= 0;
                led3 <= 0;
                led4 <= 0;
            end
            else if (counter2 < 7'b1010000) begin  // 80
                led1 <= 1;
                led2 <= 1;
                led3 <= 0;
                led4 <= 0;
            end
            else if (counter2 < 7'b1111000) begin  // 120
                led1 <= 1;
                led2 <= 1;
                led3 <= 1;
                led4 <= 0;
            end
            else if (counter2 >= 7'b1111000) begin  // 160
                led1 <= 1;
                led2 <= 1;
                led3 <= 1;
                led4 <= 1;
            end
            
            counter2 <= 0;
        end

        if (uart_ready) begin
            counter2 <= counter2 + 1;  
        end
    end
end
endmodule


module sonic(
    input clock,
    input restart,
    output trig,
    input echo,
    output reg led,
    output reg [32:0] distance
);
    reg [32:0] amcro = 0;
    reg _trig = 1'b0;
    reg [9:0] oneus = 0;
    reg [9:0] tenus = 0;
    reg [21:0] fortyms = 0;

    wire one_us = (oneus == 0);
    wire ten_us = (tenus== 0);
    wire forty_ms = (fortyms == 0);

    assign trig = _trig;

    always @(posedge clock) begin
        if (restart) begin
            amcro <= 0;
            _trig <= 1'b0;
            oneus <= 0;
            tenus <= 0;
            fortyms <= 0;
            distance <= 0;
            led <= 0;
        end else begin
            oneus <= (one_us ? 50 : oneus) - 1;
            tenus <= (ten_us ? 500 : tenus) - 1;
            fortyms <= (forty_ms ? 2000000 : fortyms) - 1;

            if (ten_us && _trig)
                _trig <= 1'b0;

            if (one_us) begin
                if (echo) begin
                    amcro <= amcro + 1;
                end else if (amcro) begin
                    distance <= amcro / 58;
                    amcro <= 0;
                end
                if(distance >10) begin
                    led<=1;                //!!!!!workkkkkk
                   end
                 else begin
                   led <=0;
                  end
            end

            if (forty_ms)
                _trig <= 1'b1;
        end
    end
endmodule


module uart_tx_module (
    input wire clk,          // System clock
    input wire reset,        // Reset signal
    input wire [7:0] data,   // 8-bit data
    input wire send,         // Signal to start
    output reg tx,           // data to send each time
    output reg busy          // Busy flag to indicate transmission in progress
);

    parameter CLOCK_FREQ = 100000000;  // 100 MHz clock frequency
    parameter BAUD_RATE = 9600;       // 9600 baud rate

    localparam BAUD_TICK_COUNT = CLOCK_FREQ / BAUD_RATE;
    localparam HALF_BAUD_TICK_COUNT = BAUD_TICK_COUNT / 2;

    reg [15:0] baud_tick_counter = 0;   
    reg [3:0] bit_index = 0;            // Index for the bits
    reg [9:0] tx_shift_reg = 10'b1111111111; 

    reg sending = 0; // Flag to indicate if sending is in progress

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            tx <= 1'b1;  // Idle state
            busy <= 1'b0;
            sending <= 1'b0;
            baud_tick_counter <= 0;
            bit_index <= 0;
            tx_shift_reg <= 10'b1111111111;
        end else begin
            if (send && !busy) begin
                tx_shift_reg <= {1'b1, data, 1'b0}; // LSB first: 1 stop, 8 data bits, 1 start
                busy <= 1'b1;
                sending <= 1'b1;
                bit_index <= 0;
                baud_tick_counter <= 0;
            end

            if (sending) begin
                if (baud_tick_counter < BAUD_TICK_COUNT - 1) begin
                    baud_tick_counter <= baud_tick_counter + 1;
                end else begin
                    baud_tick_counter <= 0;
                    tx <= tx_shift_reg[0];
                    tx_shift_reg <= {1'b1, tx_shift_reg[9:1]}; 
                    bit_index <= bit_index + 1;
                    if (bit_index == 9) begin
                        sending <= 1'b0;
                        busy <= 1'b0; // done
                        tx <= 1'b1; // Idle state
                    end
                end
            end
        end
    end
endmodule

module uart_rx_module(
    input wire clk,           
    input wire reset,         
    input wire rx,            
    output reg [7:0] data,    
    output ready          
);

    parameter CLOCK_FREQ = 100000000;  // System clock frequency
    parameter BAUD_RATE = 9600;       // UART baud rate

    localparam BAUD_TICK_COUNT = CLOCK_FREQ / BAUD_RATE;
    localparam HALF_BAUD_TICK_COUNT = BAUD_TICK_COUNT / 2;

    reg [15:0] baud_tick_counter = 0;
    reg [3:0] bit_index = 0;
    reg [9:0] rx_shift_reg = 10'b1111111111;
    reg receiving = 0;
    reg [7:0] data_reg = 0;
    reg ready = 0;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            ready <= 1'b0;
            receiving <= 1'b0;
            baud_tick_counter <= 0;
            bit_index <= 0;
            data <= 0;  
        end else if (receiving) begin
            if (baud_tick_counter < BAUD_TICK_COUNT - 1) begin
                baud_tick_counter <= baud_tick_counter + 1;
            end else begin
                baud_tick_counter <= 0;
                rx_shift_reg <= {rx, rx_shift_reg[9:1]}; 
                bit_index <= bit_index + 1;
                if (bit_index == 9) begin
                    receiving <= 1'b0;
                    ready <= 1'b1;  //data ready????????
                    data <= rx_shift_reg[8:1];  
                end
            end
        end else if (rx == 0) begin
            receiving <= 1'b1;  
            baud_tick_counter <= BAUD_TICK_COUNT / 2;  
            ready <= 1'b0;
        end else if (ready) begin
            ready <= 1'b0;  
        end
    end

endmodule


